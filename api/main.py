"""
api/main.py - thin HTTP layer over core.service. No RAG logic lives here: only validation,
auth, size limits and error mapping.

Run locally:   uvicorn api.main:app --reload      then open http://127.0.0.1:8000/docs
Optional auth: set RAG_API_KEY and send it as the `x-api-key` header (health stays open).
"""
from __future__ import annotations

import logging
import os
import secrets
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core import service

log = logging.getLogger("rag.api")
MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

app = FastAPI(title="RAG Research Assistant API", version="1.0.0")


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """If RAG_API_KEY is set, every call (except /health) must present it. Unset = open (local dev)."""
    expected = os.environ.get("RAG_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key.encode(), expected.encode())):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    cached: bool


class AskRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    doc_id: str
    answer: str
    low_confidence: bool
    sources: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Plain `def` (not async): these call blocking network APIs, so FastAPI runs them in a worker thread.
@app.post("/documents", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def upload_document(file: UploadFile = File(...)) -> dict:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)  # never read more than the limit + 1 byte
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file larger than {MAX_UPLOAD_BYTES} bytes")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="only PDF files are accepted")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.pdf"  # fixed name: the client's filename is never used as a path
        path.write_bytes(data)
        try:
            return service.ingest_pdf(path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:
            log.exception("ingest failed")
            raise HTTPException(status_code=502, detail="could not process the document")


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask_question(req: AskRequest) -> dict:
    try:
        out = service.ask(req.doc_id, req.question, top_k=req.top_k)
    except service.DocumentNotFound:
        raise HTTPException(status_code=404, detail="unknown doc_id")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        log.exception("ask failed")  # details go to the server log, never to the client
        raise HTTPException(status_code=502, detail="the language model call failed")
    return {
        "doc_id": out["doc_id"],
        "answer": out["answer"],
        "low_confidence": out["low_confidence"],
        "sources": [c[:300] for c in out["retrieved_chunks"]],
    }
