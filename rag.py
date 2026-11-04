"""Grounded Gemini answer generation with optional retrieved figures."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from google import genai
from google.genai import types


def _image_part(path: str) -> types.Part:
    image = Image.open(path).convert("RGB")
    image.thumbnail((1600, 1600))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def answer_question(
    client: genai.Client,
    question: str,
    matches: list[dict[str, Any]],
    model: str,
) -> str:
    evidence: list[str] = []
    image_parts: list[types.Part] = []

    for number, match in enumerate(matches, start=1):
        page = str(match["page_start"])
        if match["page_end"] != match["page_start"]:
            page += f"-{match['page_end']}"
        evidence.append(
            f"[Source {number} | PDF page {page} | {match['section']} | {match['kind']}]\n{match['text']}"
        )
        image_path = match.get("image_path")
        if image_path and Path(str(image_path)).exists() and len(image_parts) < 2:
            image_parts.append(_image_part(str(image_path)))

    prompt = (
        "Answer using only the supplied FortiAnalyzer 7.0.0 manual excerpts and images. "
        "Reply in the user's language. Preserve CLI commands exactly. Cite each factual claim "
        "with the source label, for example [Source 2, PDF p. 42]. If the evidence does not "
        "answer the question, say that the manual excerpts do not establish the answer.\n\n"
        f"Question:\n{question}\n\nEvidence:\n" + "\n\n".join(evidence)
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt, *image_parts],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text or "Gemini returned an empty answer."