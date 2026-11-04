from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag import answer_question
from vector_store import DEFAULT_EMBEDDING_MODEL, GeminiVectorStore


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "Administration_Guide.pdf"
DATABASE_PATH = DATA_DIR / "index.sqlite3"
IMAGE_DIR = DATA_DIR / "images"

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Manual RAG", page_icon="📘", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink: #172321; --muted: #64716e; --accent: #c1492e; --line: #dce3df; }
    .stApp { background: #f7f8f5; color: var(--ink); }
    [data-testid="stSidebar"] { background: #edf1ed; border-right: 1px solid var(--line); }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: var(--ink); }
    .eyebrow { color: var(--accent); font-size: .75rem; font-weight: 700; letter-spacing: .12em; }
    .status-line { color: var(--muted); font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

store = GeminiVectorStore(DATABASE_PATH, IMAGE_DIR)

with st.sidebar:
    st.markdown('<div class="eyebrow">LOCAL DOCUMENT · GEMINI API</div>', unsafe_allow_html=True)
    st.header("Index settings")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        st.caption("API key loaded from the server's `.env` file.")
    else:
        st.warning("Set GEMINI_API_KEY in the server's `.env` file, then restart the app.")
    embedding_model = st.text_input(
        "Embedding model", value=os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    generation_model = st.text_input(
        "Generation model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    st.caption("Embeddings and answers use the Gemini API. Free-tier quotas and rate limits apply.")

    if not PDF_PATH.exists():
        st.error(f"PDF not found: {PDF_PATH.name}")
    else:
        status = store.index_status(PDF_PATH, embedding_model)
        if status["ready"]:
            st.markdown(
                f'<p class="status-line">Index ready · {status["chunks"]} chunks</p>',
                unsafe_allow_html=True,
            )
            if status["staged_chunks"]:
                st.markdown(
                    f'<p class="status-line">Rebuild progress saved · {status["staged_chunks"]} chunks</p>',
                    unsafe_allow_html=True,
                )
        elif status["staged_chunks"]:
            st.markdown(
                f'<p class="status-line">Saved progress · {status["staged_chunks"]} chunks; the next run will resume</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="status-line">Index not built, or the PDF / embedding model has changed</p>',
                unsafe_allow_html=True,
            )

        if st.button("Build / Rebuild Index", type="primary", use_container_width=True):
            if not api_key:
                st.error("Set GEMINI_API_KEY in the server's `.env` file, then restart the app.")
            else:
                progress = st.progress(0, text="Reading PDF and generating Gemini embeddings…")
                try:
                    client = genai.Client(api_key=api_key)
                    count = store.index_pdf(
                        client,
                        PDF_PATH,
                        embedding_model=embedding_model,
                        progress=lambda value: progress.progress(
                            value, text=f"Generating Gemini embeddings… {value:.0%}"
                        ),
                        force=status["ready"] and not status["staged_chunks"],
                    )
                    st.success(f"Indexing complete: {count} chunks")
                    st.rerun()
                except Exception as error:
                    st.error(f"Indexing failed: {error}")

st.markdown(
    '<div class="eyebrow">ADMINISTRATION GUIDE</div>',
    unsafe_allow_html=True,
)
st.title("Manual RAG")
st.caption("Ask about procedures, CLI commands, and manual figures. Answers include PDF page numbers and sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            title = f"PDF p. {source['page_start']} · {source['section']} · {source['kind']}"
            with st.expander(title):
                st.write(source["text"])
                image_path = source.get("image_path")
                if image_path and Path(str(image_path)).exists():
                    st.image(str(image_path), use_container_width=True)

question = st.chat_input("For example: How do I configure trusted hosts?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not api_key:
        st.error("Set GEMINI_API_KEY in the server's `.env` file, then restart the app.")
    elif not store.index_status(PDF_PATH, embedding_model)["ready"]:
        st.error("Build the index from the sidebar first.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            with st.chat_message("assistant"):
                with st.spinner("Searching the manual and generating an answer…"):
                    matches = store.search(
                        client,
                        question,
                        embedding_model=embedding_model,
                        top_k=5,
                    )
                    response = answer_question(client, question, matches, generation_model)
                st.markdown(response)
                for source in matches:
                    title = f"PDF p. {source['page_start']} · {source['section']} · {source['kind']}"
                    with st.expander(title):
                        st.write(source["text"])
                        image_path = source.get("image_path")
                        if image_path and Path(str(image_path)).exists():
                            st.image(str(image_path), use_container_width=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": matches}
            )
        except Exception as error:
            st.error(f"Request failed: {error}")