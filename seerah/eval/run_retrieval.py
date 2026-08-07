"""Check retrieval quality against the 304-question evaluation set, before any
generation/agent layer exists.

Two things this file gives you:

  1. An interactive mode to eyeball individual questions against all three
     retrievers - same idea as seerah.cli, but pulling from the eval set
     instead of typing your own queries.
  2. A batch mode that runs all 304 questions through vector, BM25 and hybrid
     (RRF-fused) retrieval, and checks whether each question's
     supporting_quotes actually show up in the top-k results - a quote "hits"
     if it is a substring of a retrieved chunk's text. No generation, no
     judge - purely "did retrieval find the evidence."

Why quote substring-match rather than chunk_index equality: there is no
grounding pass yet resolving each supporting_quote to an exact chunk_index in
data/chunks_contextual.json (see seerah/eval - that step hasn't been built).
A quote is guaranteed to sit entirely inside exactly one chunk, because chunks
came from a sentence-aware splitter and quotes were authored as short spans -
so "is this quote a substring of this chunk's text" is a reliable stand-in for
"is this the right chunk" without needing the grounding step first. It also
runs against the LIVE Qdrant/BM25 indexes as they exist today, not against a
frozen mapping - so a re-embed or re-chunk gets checked automatically.

Usage:
    python -m seerah.eval.run_retrieval --interactive
    python -m seerah.eval.run_retrieval --interactive --tier T3
    python -m seerah.eval.run_retrieval --interactive --id C1-001

    python -m seerah.eval.run_retrieval --batch
    python -m seerah.eval.run_retrieval --batch --top-k 5
    python -m seerah.eval.run_retrieval --batch --out data/retrieval_check.json
"""

import argparse
import json
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from seerah import config
from seerah.retrieve import Retriever

console = Console()


def load_questions():
    with open(config.DATA_DIR / "eval_questions_raw.json", encoding="utf-8") as f:
        return json.load(f)["questions"]


def filter_questions(questions, tier=None, question_id=None, cross_episode=None):
    if question_id:
        questions = [q for q in questions if q["question_id"] == question_id]
    if tier:
        questions = [q for q in questions if q["tier"] == tier]
    if cross_episode is not None:
        questions = [q for q in questions if q.get("cross_episode") == cross_episode]
    return questions


def normalize(text):
    """Collapse whitespace so a quote that wraps differently inside a chunk
    (spoken-transcript line breaks, extra spaces) still matches."""
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_hit_rank(quote, hits):
    """Returns the 1-indexed rank of the first hit containing this quote as a
    substring, or None if no hit in the list contains it."""
    needle = normalize(quote)
    for rank, hit in enumerate(hits, start=1):
        if needle in normalize(hit.text):
            return rank
    return None


# --- interactive mode --------------------------------------------------------

def print_summary_table(title, hits, hit_ranks_for_quotes=()):
    marked_ranks = set(hit_ranks_for_quotes)
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Lec", justify="right")
    table.add_column("Title", max_width=30)
    table.add_column("Chunk", justify="right")
    table.add_column("Quote?", justify="center")
    for rank, hit in enumerate(hits, start=1):
        mark = "[bold green]YES[/bold green]" if rank in marked_ranks else ""
        table.add_row(str(rank), f"{hit.score:.3f}", str(hit.lecture_number),
                      hit.canonical_title, str(hit.chunk_index), mark)
    console.print(table)


def run_interactive(questions, retriever, top_k):
    console.print(f"[bold green]Ready.[/bold green] {len(questions)} question(s) loaded. "
                  f"Press Enter to advance, 'q' to quit.\n")

    for q in questions:
        console.print(Panel(
            f"[bold]{q['question']}[/bold]\n\n[dim]tier={q['tier']}  "
            f"cross_episode={q['cross_episode']}[/dim]",
            title=q["question_id"], title_align="left",
        ))
        console.print("[dim]Reference answer:[/dim] " + q["reference_answer"])
        console.print(f"[dim]Supporting quotes ({len(q['supporting_quotes'])}):[/dim]")
        for sq in q["supporting_quotes"]:
            console.print(f"  [dim]lecture {sq['lecture_number']}:[/dim] {sq['quote'][:100]}")

        vector_hits, _, _ = retriever.vector_search(q["question"], top_k)
        bm25_hits, _ = retriever.bm25_search(q["question"], top_k)
        hybrid_hits, _ = retriever.hybrid_search(q["question"], top_k)

        v_ranks = [r for sq in q["supporting_quotes"] if (r := quote_hit_rank(sq["quote"], vector_hits))]
        b_ranks = [r for sq in q["supporting_quotes"] if (r := quote_hit_rank(sq["quote"], bm25_hits))]
        h_ranks = [r for sq in q["supporting_quotes"] if (r := quote_hit_rank(sq["quote"], hybrid_hits))]

        console.print()
        print_summary_table(f"HYBRID top-{top_k}  ({len(h_ranks)}/{len(q['supporting_quotes'])} quotes found)",
                            hybrid_hits, h_ranks)
        print_summary_table(f"VECTOR top-{top_k}  ({len(v_ranks)}/{len(q['supporting_quotes'])} quotes found)",
                            vector_hits, v_ranks)
        print_summary_table(f"BM25 top-{top_k}  ({len(b_ranks)}/{len(q['supporting_quotes'])} quotes found)",
                            bm25_hits, b_ranks)

        console.print("\n[dim]Full chunk text:[/dim]")
        for rank, hit in enumerate(hybrid_hits, start=1):
            marker = " [bold green](quote found)[/bold green]" if rank in h_ranks else ""
            console.print(Panel(hit.text, title=f"HYBRID #{rank}{marker}", title_align="left"))
        for rank, hit in enumerate(vector_hits, start=1):
            marker = " [bold green](quote found)[/bold green]" if rank in v_ranks else ""
            console.print(Panel(hit.text, title=f"VECTOR #{rank}{marker}", title_align="left"))
        for rank, hit in enumerate(bm25_hits, start=1):
            marker = " [bold green](quote found)[/bold green]" if rank in b_ranks else ""
            console.print(Panel(hit.text, title=f"BM25 #{rank}{marker}", title_align="left"))

        console.print("\n" + "=" * 90)
        try:
            if console.input("[bold cyan]Enter for next, q to quit> [/bold cyan]").strip().lower() == "q":
                break
        except (EOFError, KeyboardInterrupt):
            break


# --- batch mode ---------------------------------------------------------------

RETRIEVER_NAMES = ("vector", "bm25", "hybrid")


def score_question(q, retriever, top_k):
    vector_hits, _, _ = retriever.vector_search(q["question"], top_k)
    bm25_hits, _ = retriever.bm25_search(q["question"], top_k)
    hybrid_hits, _ = retriever.hybrid_search(q["question"], top_k)

    quotes = q["supporting_quotes"]

    def summarize(hits):
        ranks = [quote_hit_rank(sq["quote"], hits) for sq in quotes]
        found = [r for r in ranks if r is not None]
        return {
            "quotes_found": len(found),
            "quotes_total": len(ranks),
            "full_coverage": len(found) == len(ranks),
            "best_rank": min(found) if found else None,
        }

    return {
        "question_id": q["question_id"],
        "tier": q["tier"],
        "cross_episode": q["cross_episode"],
        "vector": summarize(vector_hits),
        "bm25": summarize(bm25_hits),
        "hybrid": summarize(hybrid_hits),
    }


def aggregate(results, top_k):
    by_tier = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r)
    by_tier["ALL"] = results
    by_tier["cross_episode"] = [r for r in results if r["cross_episode"]]

    summary = {}
    for tier, rows in by_tier.items():
        if not rows:
            continue
        n = len(rows)
        summary[tier] = {"n": n}
        for name in RETRIEVER_NAMES:
            summary[tier][name] = {
                f"recall@{top_k}": round(sum(r[name]["quotes_found"] / r[name]["quotes_total"] for r in rows) / n, 4),
                f"full_coverage@{top_k}": round(sum(r[name]["full_coverage"] for r in rows) / n, 4),
            }
    return summary


def run_batch(questions, retriever, top_k, out_path):
    results = []
    for i, q in enumerate(questions, start=1):
        results.append(score_question(q, retriever, top_k))
        if i % 20 == 0 or i == len(questions):
            console.print(f"  scored {i}/{len(questions)}")

    summary = aggregate(results, top_k)

    console.print(f"\n[bold]Retrieval check - top-{top_k}, {len(questions)} questions[/bold]\n")
    table = Table(show_lines=False)
    table.add_column("Tier")
    table.add_column("n", justify="right")
    for name in RETRIEVER_NAMES:
        table.add_column(f"{name} recall@{top_k}", justify="right")
        table.add_column(f"{name} full_cov@{top_k}", justify="right")
    for tier in ("T1", "T2", "T3", "cross_episode", "ALL"):
        if tier not in summary:
            continue
        s = summary[tier]
        row = [tier, str(s["n"])]
        for name in RETRIEVER_NAMES:
            row.append(f"{s[name][f'recall@{top_k}']:.3f}")
            row.append(f"{s[name][f'full_coverage@{top_k}']:.3f}")
        table.add_row(*row)
    console.print(table)

    if out_path:
        payload = {"top_k": top_k, "summary": summary, "per_question": results}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        console.print(f"\nSaved per-question detail to {out_path}")


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--interactive", action="store_true", help="step through questions one at a time")
    mode.add_argument("--batch", action="store_true", help="score all matching questions and report metrics")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tier", choices=["T1", "T2", "T3"], help="only questions in this tier")
    parser.add_argument("--id", dest="question_id", help="only this question_id (interactive mode)")
    parser.add_argument("--cross-episode", dest="cross_episode", action="store_true", default=None,
                        help="only cross_episode questions")
    parser.add_argument("--out", default=None,
                        help="batch mode: write per-question detail here (default: data/retrieval_check.json)")
    args = parser.parse_args()

    questions = load_questions()
    questions = filter_questions(questions, tier=args.tier, question_id=args.question_id,
                                 cross_episode=args.cross_episode)
    if not questions:
        console.print("[red]No questions matched the given filters.[/red]")
        return

    console.print("[bold]Loading indexes...[/bold]")
    retriever = Retriever()

    if args.interactive:
        run_interactive(questions, retriever, args.top_k)
    else:
        out_path = args.out or (config.DATA_DIR / "retrieval_check.json")
        run_batch(questions, retriever, args.top_k, out_path)


if __name__ == "__main__":
    main()
