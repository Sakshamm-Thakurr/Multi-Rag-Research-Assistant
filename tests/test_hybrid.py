import numpy as np
import pytest

import core.embeddings as emb
from core import service
from core.hybrid import bm25_ranking, rrf, tokenize


def test_identifiers_stay_whole_tokens():
    assert tokenize("What does QB-4091 mean? write-ahead log") == ["what", "does", "qb-4091", "mean", "write-ahead", "log"]


def test_bm25_puts_the_exact_identifier_first_and_ignores_non_matches():
    docs = ["QB-1001 means that thing one failed.", "QB-4091 means that quorum was lost.", "Snapshots run at night."]
    assert bm25_ranking("What does error code QB-4091 mean?", docs)[0] == 1
    assert 2 not in bm25_ranking("QB-4091", docs)  # no shared term -> not ranked at all
    assert bm25_ranking("", docs) == [] and bm25_ranking("x", []) == []


def test_rrf_rewards_agreement_between_rankings():
    # 1 and 3 both appear in both rankings (equal scores, lower index first); 2 and 9 appear once each
    assert rrf([[3, 1, 2], [1, 3, 9]]) == [1, 3, 2, 9]
    assert rrf([[5], [5]])[0] == 5
    assert rrf([[1, 2], [2, 1]]) == [1, 2]  # exact tie -> lower index first (deterministic)


class DigitBlindEmbedder:
    """Cannot tell 'QB-1001' from 'QB-4091': it ignores digits, like real embeddings blur similar numbers."""

    @staticmethod
    def _v(t):
        import re, zlib
        v = np.zeros(64, dtype="float32")
        for tok in re.findall(r"[a-z]+", t.lower()):
            v[zlib.crc32(tok.encode()) % 64] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, t):
        return self._v(t)


CHUNKS = [f"QB-{c} means that the thing failed." for c in (1001, 2002, 3003)] + ["QB-4091 means that the thing failed.",
                                                                                  "QB-5005 means that the thing failed."]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "get_embedding_model", lambda: DigitBlindEmbedder())
    st = service.LocalStore(tmp_path / "d")
    service.ingest_chunks(CHUNKS, "0123456789abcdef", st)
    return st


def test_hybrid_finds_the_exact_identifier_where_dense_cannot(store):
    q = "What does error code QB-4091 mean?"
    dense = service.retrieve("0123456789abcdef", q, top_k=1, store=store, mode="dense")["retrieved_chunks"][0]
    hybrid = service.retrieve("0123456789abcdef", q, top_k=1, store=store, mode="hybrid")["retrieved_chunks"][0]
    assert "QB-4091" not in dense  # the digit-blind embedder has no way to prefer it
    assert "QB-4091" in hybrid


def test_hybrid_result_has_the_same_shape_as_dense(store):
    out = service.retrieve("0123456789abcdef", "QB-4091", top_k=3, store=store, mode="hybrid")
    assert len(out["retrieved_chunks"]) == len(out["distances"]) == 3
    assert isinstance(out["low_confidence"], bool) and out["mode"] == "hybrid"


def test_unknown_mode_is_refused_and_default_comes_from_the_environment(store, monkeypatch):
    with pytest.raises(ValueError):
        service.retrieve("0123456789abcdef", "q", store=store, mode="magic")
    monkeypatch.setattr(service, "DEFAULT_MODE", "hybrid")
    assert service.retrieve("0123456789abcdef", "QB-4091", top_k=1, store=store)["mode"] == "hybrid"