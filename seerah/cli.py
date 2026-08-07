"""Interactive retrieval tool for the full 104-lecture corpus.

Loads both indexes once, then loops: type a question, see what each retriever
returns, with a per-query timing breakdown. This is a retrieval inspection
tool - it shows you the chunks, it does not yet generate an answer.

Usage:
    python -m seerah.cli                       # hybrid (RRF fusion) - same retrieval the agent uses
    python -m seerah.cli --top-k 5
    python -m seerah.cli --retriever vector    # vector only, skips loading BM25
    python -m seerah.cli --retriever both      # vector and BM25 shown separately, not fused
"""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from seerah import config
from seerah.retrieve import Retriever

console = Console()


def print_summary_table(title, hits):
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Lec", justify="right")
    table.add_column("Title", max_width=34)
    table.add_column("Chunk", justify="right")
    for rank, hit in enumerate(hits, start=1):
        table.add_row(str(rank), f"{hit.score:.3f}", str(hit.lecture_number),
                      hit.canonical_title, str(hit.chunk_index))
    console.print(table)


def print_full_chunks(title, hits):
    console.print(f"\n[bold underline]{title} - full chunk text[/bold underline]")
    for rank, hit in enumerate(hits, start=1):
        header = f"#{rank}  score={hit.score:.3f}  {hit.citation}  chunk {hit.chunk_index}"
        console.print(Panel(hit.text, title=header, title_align="left"))


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retriever", choices=["hybrid", "both", "vector", "bm25"], default="hybrid")
    parser.add_argument("--full-text", action="store_true", default=True,
                        help="print each retrieved chunk in full (default)")
    parser.add_argument("--no-full-text", dest="full_text", action="store_false")
    args = parser.parse_args()

    want_vector = args.retriever in ("both", "vector", "hybrid")
    want_bm25 = args.retriever in ("both", "bm25", "hybrid")

    console.print("[bold]Loading indexes (one-time cost for this session)...[/bold]")
    retriever = Retriever(load_bm25=want_bm25, load_vector=want_vector)
    console.print(
        "\n[bold green]Ready.[/bold green] Searching the full 104-lecture corpus. "
        "Type a query and press Enter. Type 'exit', 'quit', or submit an empty line to stop.\n"
    )

    while True:
        try:
            query = console.input("[bold cyan]Query> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break
        if not query or query.lower() in ("exit", "quit"):
            console.print("Bye.")
            break

        timings = []
        vector_hits = bm25_hits = hybrid_hits = None

        if args.retriever == "hybrid":
            hybrid_hits, t = retriever.hybrid_search(query, args.top_k)
            timings.append(f"vector embed = {t['embed_seconds'] * 1000:.1f} ms")
            timings.append(f"vector search = {t['vector_seconds'] * 1000:.1f} ms")
            timings.append(f"bm25 search = {t['bm25_seconds'] * 1000:.1f} ms")
            timings.append(f"RRF fuse = {t['fuse_seconds'] * 1000:.1f} ms")
        else:
            if want_vector:
                vector_hits, embed_s, search_s = retriever.vector_search(query, args.top_k)
                timings.append(f"vector embed (OpenAI API) = {embed_s * 1000:.1f} ms")
                timings.append(f"vector search = {search_s * 1000:.1f} ms")
            if want_bm25:
                bm25_hits, bm25_s = retriever.bm25_search(query, args.top_k)
                timings.append(f"bm25 search = {bm25_s * 1000:.1f} ms")

        console.print(f"\n[bold]Timing for this query:[/bold] " + "  |  ".join(timings) + "\n")

        if hybrid_hits is not None:
            print_summary_table("HYBRID (RRF fusion of vector + BM25)", hybrid_hits)
        if vector_hits is not None:
            print_summary_table("VECTOR (OpenAI large, contextual)", vector_hits)
        if bm25_hits is not None:
            print_summary_table("BM25 (contextual)", bm25_hits)

        if args.full_text:
            if hybrid_hits is not None:
                print_full_chunks("HYBRID (RRF fusion of vector + BM25)", hybrid_hits)
            if vector_hits is not None:
                print_full_chunks("VECTOR (OpenAI large, contextual)", vector_hits)
            if bm25_hits is not None:
                print_full_chunks("BM25 (contextual)", bm25_hits)

        console.print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
