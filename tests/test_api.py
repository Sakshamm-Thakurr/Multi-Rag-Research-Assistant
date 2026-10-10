"""API tests. The service layer is replaced by stubs: no network, no LLM, no cost."""
import pytest
from fastapi.testclient import TestClient

import api.main as main
from core import service

PDF = b"%PDF-1.4\n% minimal fake pdf bytes"
DOC = "0123456789abcdef"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    return TestClient(main.app)


def test_health_is_open(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_pdf_passes_the_bytes_to_ingest(client, monkeypatch):
    seen = {}

    def fake_ingest(path):
        seen["bytes"] = path.read_bytes()  # the temp file must exist while ingest runs
        seen["name"] = path.name
        return {"doc_id": DOC, "chunks": 3, "cached": False}

    monkeypatch.setattr(service, "ingest_pdf", fake_ingest)
    r = client.post("/documents", files={"file": ("../../evil.pdf", PDF, "application/pdf")})
    assert r.status_code == 200 and r.json() == {"doc_id": DOC, "chunks": 3, "cached": False}
    assert seen["bytes"] == PDF and seen["name"] == "upload.pdf"  # client filename never used


def test_non_pdf_is_rejected_with_415(client):
    r = client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_oversized_upload_is_rejected_with_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/documents", files={"file": ("a.pdf", PDF, "application/pdf")})
    assert r.status_code == 413


def test_empty_document_maps_to_422(client, monkeypatch):
    def boom(path):
        raise ValueError("no text could be extracted from the document")

    monkeypatch.setattr(service, "ingest_pdf", boom)
    r = client.post("/documents", files={"file": ("a.pdf", PDF, "application/pdf")})
    assert r.status_code == 422 and "no text" in r.json()["detail"]


def test_ask_returns_answer_and_truncated_sources(client, monkeypatch):
    monkeypatch.setattr(service, "ask", lambda doc_id, q, top_k=5: {
        "doc_id": doc_id, "answer": "42", "context_used": "c", "low_confidence": False,
        "retrieved_chunks": ["x" * 1000, "short"]})
    r = client.post("/ask", json={"doc_id": DOC, "question": "meaning?"})
    body = r.json()
    assert r.status_code == 200 and body["answer"] == "42" and body["low_confidence"] is False
    assert [len(s) for s in body["sources"]] == [300, 5]


def test_unknown_document_is_404(client, monkeypatch):
    def missing(*a, **k):
        raise service.DocumentNotFound("x")

    monkeypatch.setattr(service, "ask", missing)
    assert client.post("/ask", json={"doc_id": DOC, "question": "q"}).status_code == 404


@pytest.mark.parametrize("payload", [
    {"doc_id": "../../etc", "question": "q"},
    {"doc_id": DOC, "question": ""},
    {"doc_id": DOC, "question": "q" * 2001},
    {"doc_id": DOC, "question": "q", "top_k": 99},
    {"doc_id": DOC, "question": "q", "top_k": 0},
    {"question": "q"},
])
def test_invalid_ask_requests_are_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_llm_failure_is_502_and_leaks_nothing(client, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("401 invalid key gsk_SECRET123")

    monkeypatch.setattr(service, "ask", fail)
    r = client.post("/ask", json={"doc_id": DOC, "question": "q"})
    assert r.status_code == 502 and "gsk_" not in r.text and "SECRET" not in r.text


def test_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "s3cret")
    monkeypatch.setattr(service, "ask", lambda *a, **k: {
        "doc_id": DOC, "answer": "ok", "context_used": "", "low_confidence": False, "retrieved_chunks": []})
    c = TestClient(main.app)
    body = {"doc_id": DOC, "question": "q"}
    assert c.get("/health").status_code == 200  # health stays open
    assert c.post("/ask", json=body).status_code == 401
    assert c.post("/ask", json=body, headers={"x-api-key": "wrong"}).status_code == 401
    assert c.post("/ask", json=body, headers={"x-api-key": "s3cret"}).status_code == 200
    assert c.post("/documents", files={"file": ("a.pdf", PDF, "application/pdf")}).status_code == 401
