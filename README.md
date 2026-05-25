# Multi-Agent RAG Research Assistant

A three-agent AI system for document Q&A using Retrieval Augmented Generation (RAG).

## Architecture
PDF → Chunker → Embeddings → FAISS Index
↓
User Query → [Agent 1: Retriever] → [Agent 2: Summariser] → [Agent 3: Synthesiser] → Answer
## Agents
- **Retriever** — Converts query to vector, finds top-5 semantically similar chunks via FAISS
- **Summariser** — Compresses retrieved chunks into focused context, removes noise
- **Synthesiser** — Generates grounded, cited answer strictly from context

## Tech Stack
- **Embeddings** — Google Gemini (`gemini-embedding-001`, 3072 dimensions)
- **Vector Store** — FAISS (local, persisted to disk)
- **LLM** — Llama 3.3 70B via Groq API
- **Framework** — LangChain + Streamlit
- **PDF Parsing** — pdfplumber with RecursiveCharacterTextSplitter

## Key Features
- PDF hash check — skips re-embedding if same document uploaded again
- Low confidence detection — warns user if query doesn't match document well
- Multi-hop reasoning — handles questions requiring facts from multiple sections
- Full chat history within session

## Setup

```bash
git clone https://github.com/yourusername/rag-research-assistant
cd rag-research-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file: GOOGLE_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key


Run:
```bash
streamlit run app.py
```