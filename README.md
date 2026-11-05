# Manual RAG

A Streamlit application for searching and asking questions about the Administration Guide PDF. It uses Gemini to create embeddings and generate grounded answers, while PDF parsing, vector storage, and similarity search run locally in the Codespace.

## How it works

### Build the index

1. The app reads `data/Administration_Guide.pdf` and extracts ordered text, code blocks, page numbers, section headings, and eligible images.
2. The extracted text and code are divided into chunks. Each chunk retains its page range and section; image chunks retain a local image path.
3. Chunks are sent in batches of 16 to the Gemini Embedding API (`gemini-embedding-001` by default).
4. The resulting vectors and chunk metadata are saved to `data/index.sqlite3`. Extracted images are saved under `data/images/`.

The PDF is parsed locally; its text chunks are sent to Gemini to create embeddings. Requests are paced to stay below the free-tier 100-requests-per-minute limit, and HTTP 429 responses are retried up to five times using the API's retry delay. Each completed batch is saved in a staging table in SQLite. If indexing stops or fails, click **Build / Rebuild Index** again to resume from saved batches. The previous complete index stays available until a replacement finishes.

### Ask a question

1. The question is sent to the Gemini Embedding API to create a query vector.
2. The app compares that vector with the stored vectors locally and selects the five closest chunks.
3. The question and selected excerpts are sent to the Gemini generation model (default: `gemini-3.5-flash-lite`) to produce an answer with source/page citations.
4. If retrieved chunks have associated images, up to two images are also sent to Gemini. Images remain in the Codespace otherwise.

Each question uses the Gemini API for its query embedding and generated answer. The chunk search itself is local.

## Project structure

- `app.py` — Streamlit UI, configuration, indexing action, and chat flow.
- `pdf_ingest.py` — PDF text/code/image extraction and page-aware chunking.
- `vector_store.py` — Gemini embedding requests, 429 retries, SQLite persistence, and local cosine-similarity search.
- `rag.py` — Builds the grounded prompt, optionally attaches retrieved images, and calls the Gemini generation model.
- `tests/` — Unit tests for chunking and vector search behavior.
- `data/` — Local PDF, extracted images, and generated SQLite index. This directory is ignored by Git.

## Run in Codespaces

Python 3.11 or newer is recommended.

1. Put `Administration_Guide.pdf` in `data/`.
2. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. If you do not already have a `.env` file, create one from the example, then set your Google AI Studio API key:

   ```bash
   cp -n .env.example .env
   ```

   Set `GEMINI_API_KEY` in `.env`. Optionally set `GEMINI_EMBEDDING_MODEL` or `GEMINI_MODEL` to override the default models.

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. In the browser, click **Build / Rebuild Index** and wait for indexing to complete. Then ask questions in the chat.

The API key is loaded by the Streamlit server from `.env`; it is not shown as a browser input. Restart the app after changing `.env`. Never commit `.env` or your API key. The PDF, extracted images, and generated index are also local and ignored by Git.

Gemini API quotas, rate limits, availability, and pricing depend on your project and selected models. Check the current Gemini API rate limits and pricing before indexing or changing models.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests
```
