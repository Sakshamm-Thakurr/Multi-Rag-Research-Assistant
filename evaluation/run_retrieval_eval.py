"""
Retrieval quality gate.

    python -m evaluation.run_retrieval_eval --write-baseline    # first time: record today's scores
    python -m evaluation.run_retrieval_eval                      # later / in CI: fail if scores dropped

Measures, on a small golden set about a fictional handbook (so the LLM cannot know the answers):
  hit@k : share of questions where a retrieved chunk contains the expected phrase
  MRR   : mean of 1/rank of the first chunk that does
Retrieval only: no LLM call, so it is cheap and deterministic. Exit code 0 = ok, 1 = regression, 2 = no baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from core import service

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden.json"
BASELINES = HERE / "baselines"
REPORT = HERE / "last_report.json"
MARGIN = 0.05  # how far below today's score the saved floor sits (absorbs embedding noise)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def phrase_in(text: str, phrase: str) -> bool:
    """Whole-token match, so '5 ms' does not match inside '125 ms'. Any run of whitespace (including a
    line break that PDF extraction put in the middle of a phrase) counts as one space."""
    return re.search(rf"(?<!\w){re.escape(_squash(phrase))}(?!\w)", _squash(text), re.IGNORECASE) is not None


def first_hit_rank(chunks: list[str], phrases: list[str]):
    for rank, chunk in enumerate(chunks, 1):
        if any(phrase_in(chunk, p) for p in phrases):
            return rank
    return None


def compute_metrics(ranks: list, k: int) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "k": k,
        "hit_at_k": sum(1 for r in ranks if r is not None and r <= k) / n,
        "hit_at_1": sum(1 for r in ranks if r == 1) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def check_against_baseline(metrics: dict, baseline: dict) -> list[str]:
    problems = []
    if metrics["k"] != baseline["k"]:
        problems.append(f"k differs (run {metrics['k']}, baseline {baseline['k']}); re-record the baseline")
    if metrics["hit_at_k"] < baseline["min_hit_at_k"]:
        problems.append(f"hit@{metrics['k']} {metrics['hit_at_k']:.3f} is below the floor {baseline['min_hit_at_k']:.3f}")
    if metrics["mrr"] < baseline["min_mrr"]:
        problems.append(f"MRR {metrics['mrr']:.3f} is below the floor {baseline['min_mrr']:.3f}")
    return problems


def make_baseline(metrics: dict) -> dict:
    """Floors sit one margin below today's scores. The margin is at least ONE question's worth (1/n):
    a gate that fails when a single question flips is flaky, and a flaky gate gets ignored."""
    n = metrics["n"]
    margin = max(MARGIN, 1.0 / n)

    def floor4(x: float) -> float:  # round DOWN, so exactly one flipped question still passes
        return math.floor(max(0.0, x) * 10000) / 10000

    return {
        "k": metrics["k"],
        "n": n,
        "margin": round(margin, 4),
        "measured_hit_at_k": round(metrics["hit_at_k"], 4),
        "measured_mrr": round(metrics["mrr"], 4),
        "min_hit_at_k": floor4(metrics["hit_at_k"] - margin),
        "min_mrr": floor4(metrics["mrr"] - margin),
    }


def evaluate(k: int, golden_path: Path = GOLDEN, store=None, mode: str = "dense") -> dict:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    corpus = golden_path.parent / golden["corpus"]
    store = store or service.LocalStore(service.DATA_DIR / "eval")
    info = service.ingest_pdf(corpus, store)
    rows, ranks = [], []
    for q in golden["questions"]:
        out = service.retrieve(info["doc_id"], q["question"], top_k=k, store=store, mode=mode)
        rank = first_hit_rank(out["retrieved_chunks"], q["expected_any"])
        ranks.append(rank)
        rows.append({
            "id": q["id"], "rank": rank, "question": q["question"], "expected_any": q["expected_any"],
            "best_distance": round(out.get("best_dense_distance", out["distances"][0] if out["distances"] else 999), 3),
            "retrieved_preview": [" ".join(c.split())[:160] for c in out["retrieved_chunks"]],
        })
    return {"doc": info, "mode": mode, "golden": golden_path.name, "rows": rows, "metrics": compute_metrics(ranks, k)}


def main(argv=None) -> int:
    # The agents print emoji. When output is piped on Windows, Python falls back to cp1252 and crashes on
    # them, so force UTF-8 (and never fail on an unencodable character).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # not a real console (e.g. captured by a test runner)
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--explain", action="store_true", help="for every MISS, print what was retrieved instead")
    ap.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--baseline", type=Path, default=None, help="default: evaluation/baselines/<mode>_<golden>.json")
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args(argv)

    if args.baseline is None:
        BASELINES.mkdir(exist_ok=True)
        args.baseline = BASELINES / f"{args.mode}_{args.golden.stem}.json"
    result = evaluate(args.k, args.golden, mode=args.mode)
    m = result["metrics"]
    print("\n" + "=" * 60)
    print(f"mode: {args.mode}   golden: {args.golden.name}   documents: {result['doc']['chunks']} chunks (cached={result['doc']['cached']})")
    for r in result["rows"]:
        print(f"  {r['id']:<14} {'rank ' + str(r['rank']) if r['rank'] else 'MISS':<8} best_distance={r['best_distance']}")
    if args.explain:
        for r in (r for r in result["rows"] if r["rank"] is None):
            print(f"\nMISS {r['id']}: {r['question']}  (wanted: {r['expected_any']})")
            for i, prev in enumerate(r["retrieved_preview"], 1):
                print(f"   #{i}: {prev}")
    print(f"\nhit@{m['k']} = {m['hit_at_k']:.3f}   hit@1 = {m['hit_at_1']:.3f}   MRR = {m['mrr']:.3f}   (n = {m['n']})")
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.write_baseline:
        args.baseline.write_text(json.dumps(make_baseline(m), indent=2), encoding="utf-8")
        print(f"baseline written to {args.baseline} (floor = measured - {MARGIN})")
        return 0
    if not args.baseline.exists():
        print("no baseline found; run once with --write-baseline and commit baseline.json")
        return 2
    problems = check_against_baseline(m, json.loads(args.baseline.read_text(encoding="utf-8")))
    if problems:
        print("\nQUALITY GATE FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("\nquality gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())