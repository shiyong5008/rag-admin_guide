import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from google.genai import errors

from vector_store import (
    EMBEDDING_DIMENSIONS,
    GeminiVectorStore,
    MIN_EMBED_REQUEST_INTERVAL_SECONDS,
    _EmbeddingRequestPacer,
    _embed_content_with_retries,
)


class FakeModels:
    def embed_content(self, model, contents, config):
        self.last_task = config.task_type
        vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        vector[0 if "alpha" in contents.lower() else 1] = 1.0
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vector)])


class VectorStoreTests(unittest.TestCase):
    def test_request_pacer_waits_between_requests(self) -> None:
        pacer = _EmbeddingRequestPacer(MIN_EMBED_REQUEST_INTERVAL_SECONDS)

        with (
            patch("vector_store.time.monotonic", side_effect=[10.0, 10.0, 10.2, 10.9]),
            patch("vector_store.time.sleep") as sleep,
        ):
            pacer.wait()
            pacer.wait()

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.6)

    def test_embedding_retries_rate_limit_using_server_delay(self) -> None:
        quota_error = errors.APIError(
            429,
            {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "2s",
                        }
                    ],
                }
            },
        )
        response = SimpleNamespace(embeddings=[])
        calls = 0

        def embed():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise quota_error
            return response

        with (
            patch("vector_store._EMBED_REQUEST_PACER.wait"),
            patch("vector_store.time.sleep") as sleep,
        ):
            result = _embed_content_with_retries(embed)

        self.assertIs(result, response)
        self.assertEqual(calls, 2)
        sleep.assert_called_once_with(3.0)

    def test_embedding_does_not_retry_non_rate_limit_errors(self) -> None:
        api_error = errors.APIError(400, {"error": {"code": 400, "message": "Bad request"}})
        embed = unittest.mock.Mock(side_effect=api_error)

        with (
            patch("vector_store._EMBED_REQUEST_PACER.wait"),
            patch("vector_store.time.sleep") as sleep,
        ):
            with self.assertRaises(errors.APIError):
                _embed_content_with_retries(embed)

        embed.assert_called_once()
        sleep.assert_not_called()

    def test_indexing_resumes_from_saved_embedding_batch(self) -> None:
        class IndexModels:
            def __init__(self, fail_on_call=None):
                self.calls = []
                self.fail_on_call = fail_on_call

            def embed_content(self, model, contents, config):
                self.calls.append(contents)
                if len(self.calls) == self.fail_on_call:
                    raise errors.APIError(
                        400, {"error": {"code": 400, "message": "Test failure"}}
                    )
                vectors = []
                for content in contents:
                    vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
                    vector[0] = 1.0
                    vectors.append(SimpleNamespace(values=vector))
                return SimpleNamespace(embeddings=vectors)

        chunks = [
            {
                "text": f"Chunk {index}",
                "kind": "text",
                "page_start": index + 1,
                "page_end": index + 1,
                "section": "Test",
            }
            for index in range(3)
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_path = root / "manual.pdf"
            pdf_path.write_bytes(b"test PDF contents")
            store = GeminiVectorStore(root / "index.sqlite3", root / "images")
            first_models = IndexModels(fail_on_call=2)
            first_client = SimpleNamespace(models=first_models)

            with (
                patch("vector_store.extract_pdf_elements", return_value=[]),
                patch("vector_store.chunk_elements", return_value=chunks),
                patch("vector_store.BATCH_SIZE", 2),
                patch("vector_store._EMBED_REQUEST_PACER.wait"),
            ):
                with self.assertRaises(errors.APIError):
                    store.index_pdf(first_client, pdf_path)

                with store._connect() as connection:
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0
                    )
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM index_staging").fetchone()[0], 2
                    )

                second_models = IndexModels()
                count = store.index_pdf(SimpleNamespace(models=second_models), pdf_path)

            self.assertEqual(count, 3)
            self.assertEqual(len(second_models.calls), 1)
            self.assertEqual(second_models.calls[0], ["Section: Test\nType: text\nChunk 2"])
            self.assertEqual(store.index_status(pdf_path, "gemini-embedding-001")["chunks"], 3)
            with store._connect() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM index_staging").fetchone()[0], 0)

    def test_search_ranks_the_nearest_local_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = GeminiVectorStore(root / "index.sqlite3", root / "images")
            first_vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
            first_vector[0] = 1.0
            second_vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
            second_vector[1] = 1.0
            with store._connect() as connection:
                connection.executemany(
                    """INSERT INTO chunks
                        (text, kind, page_start, page_end, section, image_path, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        ("Alpha procedure", "text", 12, 12, "Setup", None, first_vector.tobytes()),
                        ("Beta procedure", "text", 13, 13, "Setup", None, second_vector.tobytes()),
                    ],
                )
            client = SimpleNamespace(models=FakeModels())

            results = store.search(client, "alpha procedure")

            self.assertEqual(results[0]["text"], "Alpha procedure")
            self.assertEqual(results[0]["page_start"], 12)
            self.assertEqual(client.models.last_task, "RETRIEVAL_QUERY")


if __name__ == "__main__":
    unittest.main()