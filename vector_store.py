"""Gemini embeddings stored locally in SQLite."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from google import genai
from google.genai import errors
from google.genai import types

from pdf_ingest import chunk_elements, extract_pdf_elements


DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
BATCH_SIZE = 16
MAX_EMBED_RETRIES = 5
MIN_EMBED_REQUEST_INTERVAL_SECONDS = 0.8


class _EmbeddingRequestPacer:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self._last_request_at: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                delay = self.interval_seconds - (now - self._last_request_at)
                if delay > 0:
                    time.sleep(delay)
            self._last_request_at = time.monotonic()


_EMBED_REQUEST_PACER = _EmbeddingRequestPacer(MIN_EMBED_REQUEST_INTERVAL_SECONDS)


def _retry_delay_seconds(error: errors.APIError, attempt: int) -> float:
    details = error.details
    error_body = details.get("error", details) if isinstance(details, dict) else {}
    if isinstance(error_body, dict):
        for detail in error_body.get("details", []):
            if not isinstance(detail, dict) or not str(detail.get("@type", "")).endswith("RetryInfo"):
                continue
            retry_delay = detail.get("retryDelay")
            if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                try:
                    return max(float(retry_delay[:-1]), 0.0) + 1.0
                except ValueError:
                    break
    return min(2.0 ** (attempt + 1), 60.0)


def _embed_content_with_retries(
    embed: Callable[[], types.EmbedContentResponse],
) -> types.EmbedContentResponse:
    attempt = 0
    while True:
        try:
            _EMBED_REQUEST_PACER.wait()
            return embed()
        except errors.APIError as error:
            if error.code != 429 or attempt >= MAX_EMBED_RETRIES:
                raise
            time.sleep(_retry_delay_seconds(error, attempt))
            attempt += 1


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GeminiVectorStore:
    def __init__(self, database_path: str | Path, image_dir: str | Path) -> None:
        self.database_path = Path(database_path)
        self.image_dir = Path(image_dir)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    image_path TEXT,
                    embedding BLOB NOT NULL
                )"""
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS index_staging (
                    chunk_index INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    image_path TEXT,
                    embedding BLOB NOT NULL
                )"""
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS staging_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def index_status(self, pdf_path: str | Path, embedding_model: str) -> dict[str, int | bool]:
        if not Path(pdf_path).exists():
            return {"ready": False, "chunks": 0, "staged_chunks": 0}
        source_hash = file_hash(pdf_path)
        with self._connect() as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
            count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            staging_metadata = dict(
                connection.execute("SELECT key, value FROM staging_metadata").fetchall()
            )
            staged_count = connection.execute("SELECT COUNT(*) FROM index_staging").fetchone()[0]
        ready = (
            count > 0
            and metadata.get("source_hash") == source_hash
            and metadata.get("embedding_model") == embedding_model
        )
        staging_matches = (
            staging_metadata.get("source_hash") == source_hash
            and staging_metadata.get("embedding_model") == embedding_model
        )
        return {
            "ready": ready,
            "chunks": count if ready else 0,
            "staged_chunks": staged_count if staging_matches else 0,
        }

    def index_pdf(
        self,
        client: genai.Client,
        pdf_path: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        progress: Callable[[float], None] | None = None,
        force: bool = False,
    ) -> int:
        source_hash = file_hash(pdf_path)
        with self._connect() as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
            existing_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if (
            not force
            and existing_count > 0
            and metadata.get("source_hash") == source_hash
            and metadata.get("embedding_model") == embedding_model
        ):
            return existing_count

        elements = extract_pdf_elements(pdf_path, self.image_dir)
        chunks = chunk_elements(elements)
        if not chunks:
            raise ValueError("No extractable text or images were found in this PDF.")

        documents = [
            f"Section: {chunk['section']}\nType: {chunk['kind']}\n{chunk['text']}"
            for chunk in chunks
        ]
        with self._connect() as connection:
            staging_metadata = dict(
                connection.execute("SELECT key, value FROM staging_metadata").fetchall()
            )
            staging_matches = (
                staging_metadata.get("source_hash") == source_hash
                and staging_metadata.get("embedding_model") == embedding_model
            )
            if force or not staging_matches:
                connection.execute("DELETE FROM index_staging")
                connection.execute("DELETE FROM staging_metadata")
                connection.executemany(
                    "INSERT INTO staging_metadata (key, value) VALUES (?, ?)",
                    [
                        ("source_hash", source_hash),
                        ("embedding_model", embedding_model),
                    ],
                )
            staged_count, last_staged_index = connection.execute(
                "SELECT COUNT(*), COALESCE(MAX(chunk_index), -1) FROM index_staging"
            ).fetchone()
            if staged_count != last_staged_index + 1 or staged_count > len(chunks):
                connection.execute("DELETE FROM index_staging")
                staged_count = 0

        if progress:
            progress(staged_count / len(documents))

        for start in range(staged_count, len(documents), BATCH_SIZE):
            batch = documents[start : start + BATCH_SIZE]
            response = _embed_content_with_retries(
                lambda: client.models.embed_content(
                    model=embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=EMBEDDING_DIMENSIONS,
                    ),
                ),
            )
            if not response.embeddings or len(response.embeddings) != len(batch):
                raise RuntimeError("Gemini returned an incomplete embedding batch.")
            batch_vectors = [
                np.asarray(item.values, dtype=np.float32) for item in response.embeddings
            ]
            with self._connect() as connection:
                connection.executemany(
                    """INSERT INTO index_staging
                        (chunk_index, text, kind, page_start, page_end, section, image_path, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            start + offset,
                            chunk["text"],
                            chunk["kind"],
                            chunk["page_start"],
                            chunk["page_end"],
                            chunk["section"],
                            chunk.get("image_path"),
                            vector.tobytes(),
                        )
                        for offset, (chunk, vector) in enumerate(
                            zip(chunks[start : start + len(batch)], batch_vectors, strict=True)
                        )
                    ],
                )
            if progress:
                progress(min(1.0, (start + len(batch)) / len(documents)))

        with self._connect() as connection:
            staged_rows = connection.execute(
                """SELECT text, kind, page_start, page_end, section, image_path, embedding
                    FROM index_staging ORDER BY chunk_index"""
            ).fetchall()
            if len(staged_rows) != len(chunks):
                raise RuntimeError("The staged embedding count does not match the extracted chunks.")
            connection.execute("DELETE FROM chunks")
            connection.executemany(
                """INSERT INTO chunks
                    (text, kind, page_start, page_end, section, image_path, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [tuple(row) for row in staged_rows],
            )
            connection.execute("DELETE FROM metadata")
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [
                    ("source_hash", source_hash),
                    ("embedding_model", embedding_model),
                ],
            )
            connection.execute("DELETE FROM index_staging")
            connection.execute("DELETE FROM staging_metadata")
        return len(chunks)

    def search(
        self,
        client: genai.Client,
        query: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        response = _embed_content_with_retries(
            lambda: client.models.embed_content(
                model=embedding_model,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            ),
        )
        if not response.embeddings:
            raise RuntimeError("Gemini did not return a query embedding.")
        query_vector = np.asarray(response.embeddings[0].values, dtype=np.float32)

        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM chunks").fetchall()
        if not rows:
            return []

        matrix = np.vstack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
        similarities = matrix @ query_vector / np.maximum(norms, 1e-8)
        ranked = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "text": rows[int(index)]["text"],
                "kind": rows[int(index)]["kind"],
                "page_start": rows[int(index)]["page_start"],
                "page_end": rows[int(index)]["page_end"],
                "section": rows[int(index)]["section"],
                "image_path": rows[int(index)]["image_path"],
                "score": float(similarities[index]),
            }
            for index in ranked
        ]