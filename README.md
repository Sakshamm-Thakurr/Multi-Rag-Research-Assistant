# Multi-Agent RAG Research Assistant

[![CI](https://github.com/Sakshamm-Thakurr/Multi-Rag-Research-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Sakshamm-Thakurr/Multi-Rag-Research-Assistant/actions/workflows/ci.yml)

Upload a PDF, ask questions, and get answers grounded in the document, with the source passages returned
alongside. A three-agent pipeline (Retriever, Summariser, Synthesiser) runs on FAISS vector search combined with
BM25 keyword search, behind both a Streamlit UI and a Dockerised FastAPI service. Retrieval quality is measured
against golden question sets, and a CI job fails the build if it gets worse.

## How it works

```
PDF -> chunks -> Gemini embeddings -> FAISS index (one folder per document, keyed by a content hash)

question -> Retriever agent : dense (FAISS) + keyword (BM25), merged with reciprocal rank fusion
         -> Summariser agent: compresses the retrieved chunks into focused context
         -> Synthesiser agent: answers using only that context; flags low-confidence retrievals
```

- **Embeddings:** Google Gemini (`gemini-embedding-001`) 3072-dimensional vectors. **Generation:** Groq-hosted `openai/gpt-oss-120b`.
  Both names are set in code (`core/embeddings.py`, `agents/summariser.py`, `agents/synthesiser.py`);
  providers retire models, which happened once during development.
- **Two front ends, one core:** `app.py` (Streamlit, original single-document flow) and `api/main.py` (FastAPI,
  per-document indexes via `core/service.py`).

## Tech stack

| Part | Choice |
|---|---|
| Embeddings | Google Gemini `gemini-embedding-001` |(3072 dimensions)
| Vector store | FAISS (local, persisted to disk, one folder per document) |
| Keyword search | BM25 (plain Python, `core/hybrid.py`) merged with vector search by reciprocal rank fusion |
| LLM | `openai/gpt-oss-120b` via the Groq API (the original `llama-3.3-70b-versatile` was retired by the provider) |
| Framework | LangChain, Streamlit (UI), FastAPI (service) |
| PDF parsing | pdfplumber with `RecursiveCharacterTextSplitter` |
| Delivery | Docker, GitHub Actions, pytest |

## Features

- **Content-hash check:** the same PDF always maps to the same document id, so it is never embedded twice.
- **Low-confidence detection:** the pipeline flags questions that do not match the document well.
- **Chat history** within a session (Streamlit app).
- **Source passages** returned with every answer (API).

## Setup

```bash
git clone https://github.com/Sakshamm-Thakurr/Multi-Rag-Research-Assistant.git
cd Multi-Rag-Research-Assistant
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (never commit it):

```
GOOGLE_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
```

```bash
streamlit run app.py
```

## Retrieval evaluation

Retrieval is scored on golden questions about a **fictional product handbook** written for this repo
(`evaluation/corpus/`). Its facts exist nowhere else, so a model cannot answer from memory, and there is no
third-party copyright. A question counts as a hit if a retrieved chunk contains the expected phrase.

| Question set | Dense: hit@3 / MRR | Hybrid (default): hit@3 / MRR |
|---|---|---|
| Development, 22 questions | 0.864 / 0.841 | 0.955 / 0.909 |
| **Held-out**, 14 questions | 0.857 / 0.786 | 1.000 / 0.929 |

How to read this honestly:
- The development set was used to diagnose problems. The held-out set was written **before** hybrid retrieval was
  measured, so it is the fairer check.
- Hybrid scored higher on every metric on both sets (35/36 vs 31/36 questions combined), but with 36 questions the
  gap is **not statistically significant** (Fisher exact p about 0.2). The case for hybrid rests on a diagnosed
  mechanism: dense embeddings return near-identical `QB-xxxx` lines for an exact error code, while BM25 matches the
  exact token (`tests/test_hybrid.py` demonstrates this).
- This is a small regression gate, not a benchmark.

```bash
python -m evaluation.run_retrieval_eval --mode hybrid                                  # score the dev set
python -m evaluation.run_retrieval_eval --mode hybrid --golden evaluation/golden_heldout.json
python -m evaluation.run_retrieval_eval --mode hybrid --explain                        # show what was retrieved for misses
python -m evaluation.run_retrieval_eval --mode hybrid --write-baseline                 # record new floors
```

Floors in `evaluation/baselines/` sit one margin below the measured scores, and the margin is at least one
question's worth, so a single flipped question does not fail the build (a flaky gate gets ignored).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check (always open) |
| `POST /documents` | Upload a PDF, returns `doc_id` (same file content gives the same id, no re-embedding) |
| `POST /ask` | `{"doc_id", "question", "top_k"}` returns the answer, `low_confidence`, and source chunks |

Built in: PDF-only uploads with a size limit, strict `doc_id` format (no path traversal), optional API key
(`x-api-key`), and provider errors logged server-side but never returned to the caller.

```bash
uvicorn api.main:app --reload            # then open http://127.0.0.1:8000/docs
```

## Run it

```bash
# API
pip install -r requirements-core.txt -r requirements-api.txt
uvicorn api.main:app --reload

# Docker (keys come in at runtime; they are never baked into the image)
docker build -t rag-api:dev .
docker run -d --name rag-api -p 127.0.0.1:8000:8000 --env-file .env \
  -e RAG_API_KEY=choose-a-key -v ragdata:/data rag-api:dev

# Tests (49, no network, no API keys)
pip install -r requirements-dev.txt && python -m pytest -q
```

| Variable | Meaning |
|---|---|
| `GOOGLE_API_KEY`, `GROQ_API_KEY` | Provider keys (never commit `.env`) |
| `RAG_API_KEY` | If set, `/documents` and `/ask` require this in `x-api-key` |
| `RAG_DATA_DIR` | Where per-document indexes are stored (default `data`; `/data` in the container) |
| `RAG_RETRIEVAL_MODE` | `hybrid` (default) or `dense` |
| `RAG_MAX_UPLOAD_BYTES` | Upload limit (default 10 MB) |

The image runs as a non-root user, has a health check, and excludes `.env`, tests and sample data
(660 MB on disk, 147 MB compressed).

## CI

`.github/workflows/ci.yml` runs on every push and pull request, and nightly:
1. **tests:** all unit tests.
2. **docker:** builds the image, starts it, checks `/health`, checks that a call without a key gets `401`, and that
   the container is non-root with no `.env` inside.
3. **retrieval-gate:** runs the quality gate with real embeddings on both question sets. It needs a
   `GOOGLE_API_KEY` repository secret and is skipped (with a warning) without it; it is never run with secrets on
   pull requests from forks.

## Known limitations

- **The chunker can cut a sentence in two.** In the evaluation, "wait at least 5" and "minutes between nodes"
  landed in different chunks, so one golden question still misses. Not fixed yet.
- **Tiny evaluation corpus:** about 20 chunks and 36 questions.
- **Hybrid is not better on every query.** For "what is an index" the best passage ranks third behind chunks that
  repeat the word "index"; the answer was still correct.
- **BM25 is recomputed for each query.** Fine for documents of tens to hundreds of chunks; it would need caching
  for large ones.
- **The Streamlit app still uses the original single-document cache;** only the API uses per-document indexes.
- **Text PDFs only** (no OCR). Model names are hard-coded.
- **Not deployed to any cloud provider.** The service is container-ready and tested in CI, nothing more.

## Layout

```
app.py              Streamlit UI (original flow)
api/main.py         FastAPI service
core/               loading, embeddings, per-document service (service.py), BM25 + RRF (hybrid.py)
agents/             retriever, summariser, synthesiser
evaluation/         golden sets, handbook PDF, quality gate, saved baselines
tests/              49 tests
Dockerfile          non-root image with health check
.github/workflows/  CI
```
