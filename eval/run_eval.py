"""RAG evaluation harness for the Resume RAG Assistant.

Scores the pipeline against eval/golden_set.json on two levels:

  Retrieval  - Hit@1, Hit@k and MRR: does the right resume appear near the top?
  Guardrails - status accuracy, plus refusal precision/recall: does the app
               refuse exactly when it should (no_match / unclear) and answer
               when it should?

Runs in whichever mode .env selects. DEMO_MODE=1 needs no API key.

    python -m eval.run_eval                  # print report
    python -m eval.run_eval --min-status-accuracy 0.9   # fail (exit 1) below a quality gate
    python -m eval.run_eval --json eval/report.json     # also write machine-readable results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.rag_chain import answer_query, is_query_too_vague, load_vectorstore  # noqa: E402

GOLDEN = Path(__file__).with_name("golden_set.json")


def ranked_sources(vs, query: str, k: int) -> list[str]:
    """Unique source files in retrieval order (answer_query sorts them alphabetically)."""
    seen: list[str] = []
    for doc, _ in vs.similarity_search_with_relevance_scores(query, k=k):
        src = doc.metadata.get("source")
        if src not in seen:
            seen.append(src)
    return seen


def evaluate(k: int) -> dict:
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    vs = load_vectorstore()
    rows = []
    for c in cases:
        result = answer_query(c["query"], vectorstore=vs)
        row = {"id": c["id"], "query": c["query"], "expected": c["expected_status"], "actual": result["status"]}
        if c["expected_status"] == "matched" and not is_query_too_vague(c["query"]):
            ranked = ranked_sources(vs, c["query"], k)
            rank = next((i + 1 for i, s in enumerate(ranked) if s in c["relevant"]), None)
            row.update(top=ranked[:1], rank=rank)
        rows.append(row)

    retrieval = [r for r in rows if "rank" in r]
    n_ret = len(retrieval) or 1
    should_refuse = [r for r in rows if r["expected"] != "matched"]
    did_refuse = [r for r in rows if r["actual"] != "matched"]
    correct_refusals = [r for r in should_refuse if r["actual"] != "matched"]

    metrics = {
        "mode": "demo" if config.DEMO_MODE else "openai",
        "cases": len(rows),
        "hit@1": sum(r["rank"] == 1 for r in retrieval) / n_ret,
        f"hit@{k}": sum(r["rank"] is not None for r in retrieval) / n_ret,
        "mrr": sum(1 / r["rank"] for r in retrieval if r["rank"]) / n_ret,
        "status_accuracy": sum(r["expected"] == r["actual"] for r in rows) / len(rows),
        "refusal_precision": len(correct_refusals) / (len(did_refuse) or 1),
        "refusal_recall": len(correct_refusals) / (len(should_refuse) or 1),
    }
    return {"metrics": metrics, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, default=config.TOP_K, help="retrieval depth (default: TOP_K)")
    ap.add_argument("--min-status-accuracy", type=float, default=None, help="exit 1 if status accuracy is below this")
    ap.add_argument("--min-hit-at-k", type=float, default=None, help="exit 1 if Hit@k is below this")
    ap.add_argument("--json", type=Path, default=None, help="write full results to this file")
    args = ap.parse_args()

    report = evaluate(args.k)
    m = report["metrics"]

    print(f"\nResume RAG evaluation ({m['mode']} mode, {m['cases']} cases)\n")
    print(f"{'ID':<11} {'EXPECTED':<9} {'ACTUAL':<9} {'RANK':<5} RESULT")
    for r in report["rows"]:
        ok = r["expected"] == r["actual"] and r.get("rank", 1) is not None
        print(f"{r['id']:<11} {r['expected']:<9} {r['actual']:<9} {str(r.get('rank', '-')):<5} {'PASS' if ok else 'FAIL'}")
    print()
    for key, val in m.items():
        if isinstance(val, float):
            print(f"  {key:<18} {val:.2f}")
    print()

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed = []
    if args.min_status_accuracy is not None and m["status_accuracy"] < args.min_status_accuracy:
        failed.append(f"status_accuracy {m['status_accuracy']:.2f} < {args.min_status_accuracy}")
    hk = m[f"hit@{args.k}"]
    if args.min_hit_at_k is not None and hk < args.min_hit_at_k:
        failed.append(f"hit@{args.k} {hk:.2f} < {args.min_hit_at_k}")
    for f in failed:
        print(f"QUALITY GATE FAILED: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
