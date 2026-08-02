"""
Interactive query tool for the FULL 104-lecture corpus: loads everything
once, then lets you type as many queries as you want in a loop, each
showing full results + a per-query timing breakdown, for both retrievers:

    - Vector: OpenAI text-embedding-3-large (seerah_full_corpus_contextual)
    - BM25: full_corpus_bm25_index

(Originally targeted the 10-lecture pilot's local BGE-M3 collection - now
points at the full-corpus OpenAI collection and full-corpus BM25 index
built in scripts/17 and scripts/18.)

Usage: python scripts/13_interactive_query_test.py
       type a query, press Enter, see results + timing
       type 'exit' or 'quit' (or just press Enter on empty input) to stop
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from llama_index.retrievers.bm25 import BM25Retriever
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
openai_client = OpenAI()

REPO_ROOT = Path(__file__).resolve().parent.parent
BM25_DIR = REPO_ROOT / "full_corpus_bm25_index"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "text-embedding-3-large"
COLLECTION_NAME = "seerah_full_corpus_contextual"
TOP_K = 10

console = Console()


def vector_search(qdrant, query):
    t0 = time.perf_counter()
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_vector = response.data[0].embedding
    embed_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )
    search_time = time.perf_counter() - t0

    results = [
        (h.score, h.payload["lecture_number"], h.payload["canonical_title"],
         h.payload["chunk_index"], h.payload["text"])
        for h in result.points
    ]
    return results, embed_time, search_time


def bm25_search(retriever, query):
    t0 = time.perf_counter()
    nodes = retriever.retrieve(query)[:TOP_K]
    search_time = time.perf_counter() - t0

    results = [
        (n.score, n.metadata["lecture_number"], n.metadata["canonical_title"],
         n.metadata["chunk_index"], n.get_content())
        for n in nodes
    ]
    return results, search_time


def print_summary_table(title, results):
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Lec", justify="right")
    table.add_column("Title", max_width=30)
    table.add_column("Chunk", justify="right")

    for rank, (score, lecture_number, canonical_title, chunk_index, _) in enumerate(results, start=1):
        table.add_row(str(rank), f"{score:.3f}", str(lecture_number), canonical_title, str(chunk_index))

    console.print(table)


def print_full_chunks(title, results):
    console.print(f"\n[bold underline]{title} - full chunk text[/bold underline]")
    for rank, (score, lecture_number, canonical_title, chunk_index, text) in enumerate(results, start=1):
        header = f"#{rank}  score={score:.3f}  lecture {lecture_number} ({canonical_title})  chunk {chunk_index}"
        console.print(Panel(text, title=header, title_align="left"))


def main():
    console.print("[bold]Loading BM25 index (one-time cost for this whole session)...[/bold]")

    qdrant = QdrantClient(url=QDRANT_URL, timeout=30)

    t0 = time.perf_counter()
    bm25_retriever = BM25Retriever.from_persist_dir(str(BM25_DIR))
    bm25_retriever.similarity_top_k = TOP_K
    console.print(f"  loaded BM25 index in {time.perf_counter()-t0:.2f}s")

    console.print("\n[bold green]Ready.[/bold green] Searching the full 104-lecture corpus. "
                  "Type a query and press Enter. "
                  "Type 'exit', 'quit', or just press Enter on empty input to stop.\n")

    while True:
        query = console.input("[bold cyan]Query> [/bold cyan]").strip()
        if not query or query.lower() in ("exit", "quit"):
            console.print("Bye.")
            break

        vector_results, embed_time, vsearch_time = vector_search(qdrant, query)
        bm25_results, bsearch_time = bm25_search(bm25_retriever, query)

        console.print(
            f"\n[bold]Timing for this query:[/bold] "
            f"vector embed (OpenAI API) = {embed_time*1000:.1f} ms  |  "
            f"vector search = {vsearch_time*1000:.1f} ms  |  "
            f"bm25 search = {bsearch_time*1000:.1f} ms\n"
        )

        print_summary_table(f"VECTOR (OpenAI large, contextual) - embed {embed_time*1000:.0f}ms + search {vsearch_time*1000:.0f}ms", vector_results)
        print_summary_table(f"BM25 (contextual) - search {bsearch_time*1000:.0f}ms", bm25_results)

        print_full_chunks("VECTOR (OpenAI large, contextual)", vector_results)
        print_full_chunks("BM25 (contextual)", bm25_results)

        console.print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
