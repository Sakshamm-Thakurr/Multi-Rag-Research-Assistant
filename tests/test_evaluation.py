"""Tests for the evaluation gate itself. No network: embeddings are a deterministic fake."""
import json
import re
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pdfplumber
import pytest

import core.embeddings as emb
from core import service
from evaluation import run_retrieval_eval as ev

GOLDEN = Path(ev.GOLDEN)
HELDOUT = GOLDEN.parent / "golden_heldout.json"
CORPUS = GOLDEN.parent / "corpus" / "quillbase_handbook.pdf"


def corpus_text() -> str:
    with pdfplumber.open(CORPUS) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


# ---------- the golden data itself ----------
@pytest.mark.parametrize("path", [GOLDEN, HELDOUT])
def test_golden_file_is_well_formed(path):
    g = json.loads(path.read_text(encoding="utf-8"))
    ids = [q["id"] for q in g["questions"]]
    assert len(ids) == len(set(ids)) >= 12
    assert all(q["question"].strip() and q["expected_any"] for q in g["questions"])
    assert (path.parent / g["corpus"]).exists()


def test_the_two_sets_do_not_overlap():
    a = {q["question"] for q in json.loads(GOLDEN.read_text(encoding="utf-8"))["questions"]}
    b = {q["question"] for q in json.loads(HELDOUT.read_text(encoding="utf-8"))["questions"]}
    assert not (a & b)


@pytest.mark.parametrize("path", [GOLDEN, HELDOUT])
def test_every_expected_phrase_is_in_the_document_and_not_ambiguous(path):
    text = corpus_text()
    for q in json.loads(path.read_text(encoding="utf-8"))["questions"]:
        count = sum(len(re.findall(rf"(?<!\w){re.escape(p)}(?!\w)", re.sub(r"\s+", " ", text), re.I)) for p in q["expected_any"])
        assert 1 <= count <= 3, f"{q['id']}: phrase appears {count} times (must be 1-3 so a hit is meaningful)"


# ---------- metric maths ----------
def test_phrase_matching_is_token_aware():
    assert ev.phrase_in("flushed every 5 ms by default", "5 ms")
    assert not ev.phrase_in("timeout is 125 ms", "5 ms")
    assert ev.phrase_in("Uses TLS 1.3 only", "tls 1.3")
    assert ev.phrase_in("cache uses 25% of memory", "25%")
    assert ev.phrase_in("wait at least 5\nminutes between nodes", "5 minutes")  # line break inside a phrase


def test_ranks_and_metrics():
    assert ev.first_hit_rank(["a", "b 7421", "c"], ["7421"]) == 2
    assert ev.first_hit_rank(["a", "b"], ["7421"]) is None
    m = ev.compute_metrics([1, 2, None, 3], k=2)
    assert m["hit_at_k"] == 0.5 and m["hit_at_1"] == 0.25
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0 + 1 / 3) / 4)


def test_baseline_floor_and_comparison():
    base = ev.make_baseline({"k": 3, "n": 22, "hit_at_k": 0.95, "mrr": 0.90})
    assert base["min_hit_at_k"] == pytest.approx(0.90, abs=1e-3) and base["min_mrr"] == pytest.approx(0.85, abs=1e-3)
    assert ev.check_against_baseline({"k": 3, "hit_at_k": 0.91, "mrr": 0.86}, base) == []
    assert len(ev.check_against_baseline({"k": 3, "hit_at_k": 0.80, "mrr": 0.86}, base)) == 1
    assert len(ev.check_against_baseline({"k": 5, "hit_at_k": 1.0, "mrr": 1.0}, base)) == 1  # k mismatch


def test_the_floor_always_tolerates_exactly_one_flipped_question():
    for n in (14, 22, 36):
        base = ev.make_baseline({"k": 3, "n": n, "hit_at_k": 1.0, "mrr": 1.0})
        one_miss = {"k": 3, "hit_at_k": (n - 1) / n, "mrr": (n - 1) / n}
        two_miss = {"k": 3, "hit_at_k": (n - 2) / n, "mrr": (n - 2) / n}
        assert ev.check_against_baseline(one_miss, base) == [], n
        assert ev.check_against_baseline(two_miss, base) != [], n


# ---------- the gate end to end, with a fake embedder and a stand-in chunker ----------
DIM = 256


def _vec(text):
    v = np.zeros(DIM, dtype="float32")
    for tok in re.findall(r"[a-z0-9_]+", text.lower()):
        v[zlib.crc32(tok.encode()) % DIM] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class GoodEmbedder:
    def embed_documents(self, texts):
        return [_vec(t) for t in texts]

    def embed_query(self, t):
        return _vec(t)


class BrokenEmbedder(GoodEmbedder):  # simulates a bad change: every query looks the same
    def embed_query(self, t):
        return _vec("zzz unrelated")


@pytest.fixture
def env(monkeypatch, tmp_path):
    loader = types.ModuleType("core.document_loader")

    def load_and_chunk_pdf(path):  # stand-in for the real chunker: ~450-character pieces
        words, chunks, cur = corpus_text().split(), [], ""
        for w in words:
            if len(cur) + len(w) > 450:
                chunks.append(cur)
                cur = ""
            cur += w + " "
        return chunks + [cur]

    loader.load_and_chunk_pdf = load_and_chunk_pdf
    monkeypatch.setitem(sys.modules, "core.document_loader", loader)
    monkeypatch.setattr(service, "DATA_DIR", tmp_path / "data")
    return tmp_path


def run(env, *extra):
    return ev.main(["--baseline", str(env / "b.json"), "--report", str(env / "r.json"), *extra])


def test_gate_records_a_baseline_then_passes(env, monkeypatch):
    monkeypatch.setattr(emb, "get_embedding_model", lambda: GoodEmbedder())
    assert run(env) == 2  # no baseline yet
    assert run(env, "--write-baseline") == 0
    base = json.loads((env / "b.json").read_text())
    # The fake embedder is only a keyword counter; with ~20 chunks, chance level at k=3 is about 15%.
    assert base["measured_hit_at_k"] >= 0.5
    assert run(env) == 0
    assert json.loads((env / "r.json").read_text())["metrics"]["n"] >= 20


def test_gate_fails_when_retrieval_gets_worse(env, monkeypatch):
    monkeypatch.setattr(emb, "get_embedding_model", lambda: GoodEmbedder())
    assert run(env, "--write-baseline") == 0
    monkeypatch.setattr(emb, "get_embedding_model", lambda: BrokenEmbedder())
    assert run(env) == 1


def test_gate_runs_in_hybrid_mode_and_on_the_heldout_set(env, monkeypatch):
    monkeypatch.setattr(emb, "get_embedding_model", lambda: GoodEmbedder())
    assert run(env, "--mode", "hybrid", "--write-baseline") == 0
    assert run(env, "--mode", "hybrid") == 0
    assert run(env, "--golden", str(HELDOUT), "--write-baseline") == 0
    assert json.loads((env / "r.json").read_text())["golden"] == "golden_heldout.json"


def test_explain_prints_what_was_retrieved_for_misses(env, monkeypatch, capsys):
    monkeypatch.setattr(emb, "get_embedding_model", lambda: BrokenEmbedder())
    run(env, "--write-baseline", "--explain")
    out = capsys.readouterr().out
    assert "MISS " in out and "#1:" in out and "wanted:" in out