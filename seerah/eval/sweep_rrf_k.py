"""Sweep RRF's k constant against the 304-question eval set to find the value
that actually works best on this corpus, rather than assuming the textbook
default (k=60) is right here.

Fetches each question's vector + BM25 candidates ONCE (at candidate_pool
depth), then re-fuses them at every k value being tested purely in memory -
the sweep costs the same as a single retrieval pass regardless of how many k
values are tested, since only the fusion step depends on k, not the
embedding/search calls.

Usage:
    python -m seerah.eval.sweep_rrf_k
    python -m seerah.eval.sweep_rrf_k --k-values 5,10,20,30,40,60
    python -m seerah.eval.sweep_rrf_k --top-k 5
    python -m seerah.eval.sweep_rrf_k --out data/rrf_k_sweep.json   # per-question detail, not just tier tables
"""

import argparse
import json

from rich.console import Console
from rich.table import Table

from seerah import config
from seerah.retrieve import Retriever, reciprocal_rank_fusion
from seerah.eval.run_retrieval import load_questions, quote_hit_rank

console = Console()

DEFAULT_K_VALUES = [5, 10, 20, 30, 40, 60]
TIERS = ["T1", "T2", "T3", "cross_episode", "ALL"]


def score_at_k(quotes, vector_hits, bm25_hits, k, top_k):
    fused = reciprocal_rank_fusion((vector_hits, bm25_hits), k, top_k)
    ranks = [quote_hit_rank(sq["quote"], fused) for sq in quotes]
    found = [r for r in ranks if r is not None]
    return {
        "recall": len(found) / len(ranks),
        "full_coverage": len(found) == len(ranks),
    }


def rows_for_tier(per_question, tier):
    if tier == "ALL":
        return per_question
    if tier == "cross_episode":
        return [pq for pq in per_question if pq[0]["cross_episode"]]
    return [pq for pq in per_question if pq[0]["tier"] == tier]


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=config.RRF_CANDIDATE_POOL)
    parser.add_argument("--out", default=None,
                        help="write per-question, per-k detail here (e.g. data/rrf_k_sweep.json)")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    questions = load_questions()

    console.print("[bold]Loading indexes...[/bold]")
    retriever = Retriever()

    console.print(f"[bold]Fetching candidates for {len(questions)} questions "
                  f"(pool={args.candidate_pool}, fetched once regardless of how many k values)...[/bold]")
    per_question = []
    for i, q in enumerate(questions, start=1):
        vector_hits, _, _ = retriever.vector_search(q["question"], args.candidate_pool)
        bm25_hits, _ = retriever.bm25_search(q["question"], args.candidate_pool)
        per_question.append((q, vector_hits, bm25_hits))
        if i % 40 == 0 or i == len(questions):
            console.print(f"  fetched {i}/{len(questions)}")

    # per-question detail, computed once and reused for both the tier tables and --out
    per_question_detail = []
    for q, v, b in per_question:
        by_k = {k: score_at_k(q["supporting_quotes"], v, b, k, args.top_k) for k in k_values}
        per_question_detail.append({
            "question_id": q["question_id"],
            "tier": q["tier"],
            "cross_episode": q["cross_episode"],
            "by_k": {str(k): by_k[k] for k in k_values},
        })

    results = {}
    for tier in TIERS:
        rows = rows_for_tier(per_question, tier)
        if not rows:
            continue
        n = len(rows)
        results[tier] = {"n": n}
        matching_ids = {q["question_id"] for q, _, _ in rows}
        detail_rows = [d for d in per_question_detail if d["question_id"] in matching_ids]
        for k in k_values:
            scores = [d["by_k"][str(k)] for d in detail_rows]
            results[tier][k] = {
                "recall": sum(s["recall"] for s in scores) / n,
                "full_coverage": sum(s["full_coverage"] for s in scores) / n,
            }

    def print_table(metric_key, label):
        table = Table(title=f"Hybrid {label}@{args.top_k} by RRF k  (candidate_pool={args.candidate_pool})")
        table.add_column("Tier")
        table.add_column("n", justify="right")
        for k in k_values:
            table.add_column(f"k={k}", justify="right")
        for tier in TIERS:
            if tier not in results:
                continue
            r = results[tier]
            table.add_row(tier, str(r["n"]), *[f"{r[k][metric_key]:.3f}" for k in k_values])
        console.print(table)

    console.print()
    print_table("recall", "recall")
    print_table("full_coverage", "full_coverage")

    if args.out:
        payload = {
            "k_values": k_values,
            "top_k": args.top_k,
            "candidate_pool": args.candidate_pool,
            "tier_summary": {tier: {"n": r["n"], "by_k": {str(k): r[k] for k in k_values}}
                             for tier, r in results.items()},
            "per_question": per_question_detail,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        console.print(f"\nSaved per-question, per-k detail to {args.out}")


if __name__ == "__main__":
    main()
