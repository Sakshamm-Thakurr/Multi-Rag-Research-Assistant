"""
Keyword (BM25) ranking and reciprocal-rank fusion, in plain Python (no extra dependency).

Why: dense embeddings are weak at exact identifiers. Asked about "QB-4091", they return other "QB-xxxx"
lines because the numbers look alike; a keyword index matches the exact token. Hybrid search combines both.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# identifiers such as qb-4091 or write-ahead stay ONE token on both sides, so exact IDs match exactly
_TOKEN = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_ranking(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75, limit: int = 20) -> list[int]:
    """Indices of docs that share at least one term with the query, best first (Okapi BM25)."""
    q = set(tokenize(query))
    toks = [tokenize(d) for d in docs]
    n = len(toks)
    if not n or not q:
        return []
    avgdl = (sum(len(t) for t in toks) / n) or 1.0
    df: Counter = Counter()
    for t in toks:
        df.update(set(t))
    scores = []
    for t in toks:
        tf, dl, s = Counter(t), len(t), 0.0
        for term in q:
            if term in tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
    return ranked[:limit]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: score = sum of 1/(k + rank) over every ranking a document appears in."""
    score: Counter = Counter()
    for ranking in rankings:
        for pos, idx in enumerate(ranking, 1):
            score[idx] += 1.0 / (k + pos)
    return [i for i, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]