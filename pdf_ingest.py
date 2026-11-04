"""PDF extraction and layout-aware chunking for the RAG demo."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf


def _section_for_pages(document: pymupdf.Document) -> dict[int, str]:
    entries = document.get_toc(simple=True)
    section_by_page: dict[int, str] = {}
    active: list[str] = []
    cursor = 0

    for page_number in range(1, document.page_count + 1):
        while cursor < len(entries) and entries[cursor][2] <= page_number:
            level, title, _ = entries[cursor]
            active = active[: level - 1]
            active.append(title.strip())
            cursor += 1
        section_by_page[page_number] = " > ".join(active) or "Document"

    return section_by_page


def _clean_text_block(block: dict[str, Any], section: str) -> tuple[str, bool]:
    lines: list[str] = []
    mono_characters = 0
    total_characters = 0

    for line in block.get("lines", []):
        spans = line.get("spans", [])
        text = "".join(span.get("text", "") for span in spans).rstrip()
        if not text.strip():
            continue
        if text.strip() in {"Fortinet Inc.", "FortiAnalyzer 7.0.0 Administration Guide"}:
            continue
        if text.strip() == section and block["bbox"][1] < 110:
            continue
        if re.fullmatch(r"\d{1,4}", text.strip()):
            continue

        for span in spans:
            span_text = span.get("text", "")
            total_characters += len(span_text)
            font_name = span.get("font", "").lower()
            if span.get("flags", 0) & 8 or any(name in font_name for name in ("mono", "courier", "consolas")):
                mono_characters += len(span_text)
        lines.append(text)

    text = "\n".join(lines).strip()
    is_code = bool(text) and total_characters > 0 and mono_characters / total_characters >= 0.55
    if is_code:
        text = f"```text\n{text}\n```"
    return text, is_code


def extract_pdf_elements(
    pdf_path: str | Path,
    image_dir: str | Path,
    *,
    min_image_size: int = 100,
) -> list[dict[str, Any]]:
    """Extract ordered text/code/image elements with physical PDF page numbers."""
    pdf_path = Path(pdf_path)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    section_by_page = _section_for_pages(document)
    elements: list[dict[str, Any]] = []

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            section = section_by_page[page_number]
            blocks = page.get_text("dict", sort=True).get("blocks", [])
            image_index = 0

            for block_index, block in enumerate(blocks):
                if block.get("type") == 0:
                    text, is_code = _clean_text_block(block, section)
                    if text:
                        elements.append(
                            {
                                "kind": "code" if is_code else "text",
                                "text": text,
                                "page": page_number,
                                "section": section,
                            }
                        )
                elif block.get("type") == 1:
                    width = block.get("width", 0)
                    height = block.get("height", 0)
                    image_bytes = block.get("image", b"")
                    if width < min_image_size or height < min_image_size or len(image_bytes) < 3000:
                        continue

                    image_index += 1
                    extension = block.get("ext", "png")
                    image_path = image_dir / f"page_{page_number:04d}_image_{image_index:02d}.{extension}"
                    image_path.write_bytes(image_bytes)
                    nearby_text: list[str] = []
                    for neighbor_index in (block_index - 1, block_index + 1):
                        if 0 <= neighbor_index < len(blocks) and blocks[neighbor_index].get("type") == 0:
                            neighbor, _ = _clean_text_block(blocks[neighbor_index], section)
                            if neighbor:
                                nearby_text.append(neighbor[:400])
                    context = " ".join(nearby_text)
                    elements.append(
                        {
                            "kind": "image",
                            "text": f"Image in {section}. Nearby text: {context}" if context else f"Image in {section}.",
                            "page": page_number,
                            "section": section,
                            "image_path": str(image_path),
                            "image_ext": extension,
                        }
                    )
    finally:
        document.close()

    return elements


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    remaining = text.strip()
    chunks: list[str] = []

    while len(remaining) > max_chars:
        cut = remaining.rfind("\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = remaining.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[max(0, cut - overlap_chars) :].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def chunk_elements(
    elements: list[dict[str, Any]],
    *,
    max_chars: int = 3200,
    overlap_chars: int = 360,
) -> list[dict[str, Any]]:
    """Merge prose by section, while keeping code and image captions atomic."""
    chunks: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if not pending:
            return
        text = "\n\n".join(item["text"] for item in pending)
        pages = [item["page"] for item in pending]
        section = pending[0]["section"]
        for part in _split_text(text, max_chars, overlap_chars):
            chunks.append(
                {
                    "kind": "text",
                    "text": part,
                    "page_start": min(pages),
                    "page_end": max(pages),
                    "section": section,
                }
            )
        pending.clear()

    for element in elements:
        kind = element["kind"]
        if kind == "text":
            if pending and (
                pending[0]["section"] != element["section"]
                or pending[-1]["page"] != element["page"]
            ):
                flush_pending()
            pending.append(element)
            if sum(len(item["text"]) for item in pending) > max_chars:
                flush_pending()
        else:
            flush_pending()
            chunks.append(
                {
                    "kind": kind,
                    "text": element.get("text", ""),
                    "page_start": element["page"],
                    "page_end": element["page"],
                    "section": element["section"],
                    **({"image_path": element["image_path"]} if kind == "image" else {}),
                    **({"image_ext": element["image_ext"]} if kind == "image" else {}),
                }
            )

    flush_pending()
    return chunks