"""Tests for core/service.py. No network: embeddings are replaced by a deterministic fake."""
import re
import zlib

import numpy as np
import pytest

import core.embeddings as emb
from core.service import DocumentNotFound, LocalStore, ask, doc_id_for_bytes, ingest_chunks, ingest_pdf, retrieve

DIM = 64
DOC_A = [
    "MySQL is a relational database management system.",
    "To create an index use CREATE INDEX on a column.",
    "Backups are made with the mysqldump utility.",
]
DOC_B = [
    "Kubernetes schedules containers on worker nodes.",
    "A pod is the smallest deployable unit in Kubernetes.",
]


def _vec(text: str) -> list[float]:
    v = np.zeros(DIM, dtype="float32")
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[zlib.crc32(tok.encode()) % DIM] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class FakeEmbedder:
    def __init__(self):
        self.doc_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [_vec(t) for t in texts]

    def embed_query(self, text):
        return _vec(text)


@pytest.fixture
def fake(monkeypatch):
    f = FakeEmbedder()
    monkeypatch.setattr(emb, "get_embedding_model", lambda: f)
    return f


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "data")


def test_ingest_creates_files_and_second_ingest_is_cached(fake, store):
    did = doc_id_for_bytes(b"doc a")
    first = ingest_chunks(DOC_A, did, store)
    second = ingest_chunks(DOC_A, did, store)
    assert first == {"doc_id": did, "chunks": 3, "cached": False}
    assert second["cached"] is True and second["chunks"] == 3
    assert fake.doc_calls == 1  # the embedder was NOT called again
    assert {p.name for p in (store.root / did).iterdir()} == {"index.faiss", "chunks.json", "meta.json"}


def test_documents_are_isolated_from_each_other(fake, store):
    a, b = doc_id_for_bytes(b"A"), doc_id_for_bytes(b"B")
    ingest_chunks(DOC_A, a, store)
    ingest_chunks(DOC_B, b, store)
    assert "kubernetes" not in " ".join(retrieve(a, "database backups", store=store)["retrieved_chunks"]).lower()
    assert retrieve(b, "what is a pod", top_k=1, store=store)["retrieved_chunks"][0].startswith("A pod")


def test_retrieve_ranks_the_relevant_chunk_first(fake, store):
    did = doc_id_for_bytes(b"A")
    ingest_chunks(DOC_A, did, store)
    assert "mysqldump" in retrieve(did, "how do I make backups with mysqldump", top_k=1, store=store)["retrieved_chunks"][0]
    assert "CREATE INDEX" in retrieve(did, "create an index on a column", top_k=1, store=store)["retrieved_chunks"][0]


def test_chunks_roundtrip_unicode_through_json(fake, store):
    did = doc_id_for_bytes(b"u")
    chunks = ["Caf\u00e9 r\u00e9sum\u00e9 \u2014 \u4e2d\u6587 \U0001F600"]
    ingest_chunks(chunks, did, store)
    assert store.load(did)[1] == chunks


@pytest.mark.parametrize("bad", ["../etc", "..", "abc", "A" * 16, "0123456789abcdeg", "", "0123456789abcdef/../x"])
def test_invalid_doc_ids_are_rejected(store, bad):
    with pytest.raises(ValueError):
        store.exists(bad)


def test_missing_document_raises_not_found(fake, store):
    with pytest.raises(DocumentNotFound):
        retrieve("0123456789abcdef", "anything", store=store)


def test_empty_document_is_refused(fake, store):
    with pytest.raises(ValueError, match="no text"):
        ingest_chunks([], "0123456789abcdef", store)


def test_same_pdf_bytes_give_same_id_and_skip_the_loader(fake, store, tmp_path, monkeypatch):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF fake bytes")
    import sys
    import types

    calls = {"n": 0}
    loader = types.ModuleType("core.document_loader")

    def load_and_chunk_pdf(path):
        calls["n"] += 1
        return DOC_A

    loader.load_and_chunk_pdf = load_and_chunk_pdf
    monkeypatch.setitem(sys.modules, "core.document_loader", loader)
    r1 = ingest_pdf(pdf, store)
    r2 = ingest_pdf(pdf, store)
    assert r1["doc_id"] == r2["doc_id"] and r1["cached"] is False and r2["cached"] is True
    assert calls["n"] == 1


def test_ask_runs_the_three_stages_in_order(fake, store):
    did = doc_id_for_bytes(b"A")
    ingest_chunks(DOC_A, did, store)
    seen = {}

    def summarise(chunks, question):
        seen["chunks"] = chunks
        return {"compressed_context": "CONTEXT"}

    def synthesise(context, question, low_confidence):
        seen["ctx"], seen["low"] = context, low_confidence
        return {"answer": "ANSWER"}

    out = ask(did, "how do I make backups", store=store, summarise=summarise, synthesise=synthesise)
    assert out["answer"] == "ANSWER" and out["context_used"] == "CONTEXT" and out["doc_id"] == did
    assert seen["ctx"] == "CONTEXT" and seen["chunks"] and isinstance(seen["low"], bool)
