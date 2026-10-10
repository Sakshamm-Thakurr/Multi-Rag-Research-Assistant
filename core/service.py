"""
core/service.py - per-document ingest / retrieve / ask, built ON TOP of the existing modules.
Nothing existing is modified, so the Streamlit app (app.py -> core.pipeline.run_pipeline) keeps working.

Why this file exists:
  * The original cache is ONE global set of files (faiss_index.bin, chunks.pkl, pdf_hash.txt), so a second
    document overwrites the first. A service needs one index per document, addressed by an id.
  * Chunks are stored as JSON, not pickle: loading a pickle from shared storage (S3, later) can execute code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

import faiss
import numpy as np

import core.embeddings as emb  # used as a module so tests can swap get_embedding_model
from agents.retriever import retriever_agent
from core.hybrid import bm25_ranking, rrf

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
DEFAULT_MODE = os.environ.get("RAG_RETRIEVAL_MODE", "hybrid")  # "dense" | "hybrid"; a deployment setting, not a code change
_DOC_ID = re.compile(r"^[0-9a-f]{16}$")


class DocumentNotFound(Exception):
    pass


def doc_id_for_bytes(data: bytes) -> str:
    """Same file content -> same id, so re-uploading a document never re-embeds it."""
    return hashlib.sha256(data).hexdigest()[:16]


def _check_id(doc_id: str) -> str:
    # doc_id comes from API callers later; refuse anything that is not our own id format
    # so it can never be used to walk out of the data folder ("../..").
    if not isinstance(doc_id, str) or not _DOC_ID.match(doc_id):
        raise ValueError("invalid doc_id")
    return doc_id


class LocalStore:
    """One folder per document: index.faiss, chunks.json, meta.json. (An S3 store comes later.)"""

    def __init__(self, root: Path | str = DATA_DIR):
        self.root = Path(root)

    def _dir(self, doc_id: str) -> Path:
        return self.root / _check_id(doc_id)

    def exists(self, doc_id: str) -> bool:
        d = self._dir(doc_id)
        return (d / "index.faiss").exists() and (d / "chunks.json").exists()

    def save(self, doc_id: str, index, chunks: list[str], meta: Optional[dict] = None) -> None:
        d = self._dir(doc_id)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(d / "index.faiss"))
        (d / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(meta or {}), encoding="utf-8")

    def load(self, doc_id: str):
        if not self.exists(doc_id):
            raise DocumentNotFound(doc_id)
        d = self._dir(doc_id)
        index = faiss.read_index(str(d / "index.faiss"))
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        return index, chunks


def ingest_chunks(chunks: list[str], doc_id: str, store: Optional[LocalStore] = None, meta: Optional[dict] = None) -> dict:
    store = store or LocalStore()
    if store.exists(doc_id):
        index, _ = store.load(doc_id)
        return {"doc_id": doc_id, "chunks": int(index.ntotal), "cached": True}
    if not chunks:
        raise ValueError("no text could be extracted from the document")
    vectors = emb.get_embedding_model().embed_documents(chunks)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors, dtype="float32"))
    store.save(doc_id, index, chunks, meta)
    return {"doc_id": doc_id, "chunks": len(chunks), "cached": False}


def ingest_pdf(pdf_path: str | Path, store: Optional[LocalStore] = None) -> dict:
    """Index a PDF once; the same file content always maps to the same doc_id."""
    pdf_path = Path(pdf_path)
    store = store or LocalStore()
    doc_id = doc_id_for_bytes(pdf_path.read_bytes())
    if store.exists(doc_id):
        return ingest_chunks([], doc_id, store)  # cached path; never touches the loader or the embedder
    from core.document_loader import load_and_chunk_pdf  # imported here: heavy, and not needed when cached

    chunks = load_and_chunk_pdf(str(pdf_path))
    return ingest_chunks(chunks, doc_id, store, meta={"source": pdf_path.name})


def retrieve(doc_id: str, question: str, top_k: int = 5, store: Optional[LocalStore] = None, mode: Optional[str] = None) -> dict:
    """Retrieval only (no LLM call): cheap and deterministic, so it is what the CI quality gate uses.

    mode "dense"  : embeddings only (the original behaviour).
    mode "hybrid" : embeddings + BM25 keyword ranking, merged with reciprocal rank fusion."""
    mode = mode or DEFAULT_MODE
    if mode not in ("dense", "hybrid"):
        raise ValueError(f"unknown retrieval mode: {mode!r}")
    store = store or LocalStore()
    index, chunks = store.load(doc_id)
    if mode == "dense":
        return retriever_agent(question, index, chunks, top_k=top_k)

    pool = min(len(chunks), max(top_k * 4, 20))
    dense = emb.search_similar_chunks(question, index, chunks, top_k=pool)
    dense_rank = [r["chunk_index"] for r in dense]
    dist = {r["chunk_index"]: r["distance"] for r in dense}
    fused = rrf([dense_rank, bm25_ranking(question, chunks, limit=pool)])[:top_k]
    worst = max(dist.values()) if dist else 999.0
    best = dense[0]["distance"] if dense else 999.0
    return {
        "query": question,
        "mode": "hybrid",
        "retrieved_chunks": [chunks[i] for i in fused],
        "distances": [dist.get(i, worst) for i in fused],  # chunks found only by keywords get the pool's worst distance
        "best_dense_distance": best,
        "low_confidence": best > 1.5,  # same rule as the retriever agent, judged on the dense search
    }


def ask(
    doc_id: str,
    question: str,
    top_k: int = 5,
    store: Optional[LocalStore] = None,
    summarise: Optional[Callable] = None,
    synthesise: Optional[Callable] = None,
    mode: Optional[str] = None,
) -> dict:
    """Same three-agent flow as core.pipeline.run_pipeline, but on an already-indexed document."""
    if summarise is None:
        from agents.summariser import summariser_agent as summarise
    if synthesise is None:
        from agents.synthesiser import synthesiser_agent as synthesise
    r = retrieve(doc_id, question, top_k, store, mode)
    s = summarise(r["retrieved_chunks"], question)
    f = synthesise(s["compressed_context"], question, low_confidence=r["low_confidence"])
    return {
        "doc_id": doc_id,
        "query": question,
        "answer": f["answer"],
        "context_used": s["compressed_context"],
        "retrieved_chunks": r["retrieved_chunks"],
        "low_confidence": r["low_confidence"],
    }