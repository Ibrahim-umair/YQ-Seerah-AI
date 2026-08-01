"""
Manual, eyeball-it-yourself query tool. Edit QUERY below, run the script,
and see the top 10 results from both retrievers: a compact summary table
first (rank/score/lecture/chunk), then the FULL text of each chunk printed
below it so you can actually read and evaluate the content, not just a
truncated preview.

Only shows the "contextual" chunks - the plain (no-context) variant already
did its job proving contextual retrieval was the better choice (see the
README's Retrieval Evaluation section); this tool is for testing the chosen
approach going forward, not re-litigating that comparison. The plain Qdrant
collection and BM25 index are left untouched on disk, since they're still
what retrieval_eval_results.json depends on for reproducibility.

Also times model loading and each search call, since BGE-M3 (a real neural
model, running on CPU here) and BM25 (pure lexical, no model) have very
different cost profiles worth seeing directly.

Usage: edit QUERY, then `python scripts/12_manual_query_test.py`
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ------------------------------------------------------------------
QUERY = "why did the Quraysh want to fight at Uhud"
TOP_K = 10
# ------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from llama_index.retrievers.bm25 import BM25Retriever
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

REPO_ROOT = Path(__file__).resolve().parent.parent
BM25_DIR = REPO_ROOT / "bm25_indexes"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "recursive_contextual"

console = Console()


def vector_search(client, model):
    query_vector = model.encode([QUERY], normalize_embeddings=True)[0]
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=TOP_K,
        with_payload=True,
    )
    return [
        (h.score, h.payload["lecture_number"], h.payload["canonical_title"],
         h.payload["chunk_index"], h.payload["text"])
        for h in response.points
    ]


def bm25_search(retriever):
    nodes = retriever.retrieve(QUERY)[:TOP_K]
    return [
        (n.score, n.metadata["lecture_number"], n.metadata["canonical_title"],
         n.metadata["chunk_index"], n.get_content())
        for n in nodes
    ]


def print_summary_table(title, results, elapsed):
    table = Table(title=f"{title}  ({elapsed*1000:.0f} ms)", show_lines=False)
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
    console.print(f"\n[bold]QUERY:[/bold] {QUERY!r}\n")

    t0 = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL)
    model_load_time = time.perf_counter() - t0
    console.print(f"Loaded {EMBEDDING_MODEL} in {model_load_time:.2f}s (one-time cost, not per-query)\n")

    qdrant = QdrantClient(url=QDRANT_URL)

    t0 = time.perf_counter()
    bm25_retriever = BM25Retriever.from_persist_dir(str(BM25_DIR / COLLECTION_NAME))
    bm25_retriever.similarity_top_k = TOP_K
    bm25_load_time = time.perf_counter() - t0
    console.print(f"Loaded BM25 index in {bm25_load_time:.2f}s (one-time cost, not per-query)\n")

    t0 = time.perf_counter()
    vector_results = vector_search(qdrant, model)
    vector_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_results = bm25_search(bm25_retriever)
    bm25_time = time.perf_counter() - t0

    console.print(f"[bold]Per-query search time:[/bold] vector (BGE-M3) = {vector_time*1000:.1f} ms   |   BM25 = {bm25_time*1000:.1f} ms   "
                  f"({vector_time/bm25_time:.0f}x slower)\n" if bm25_time > 0 else "")

    print_summary_table("VECTOR (BGE-M3, contextual)", vector_results, vector_time)
    print_summary_table("BM25 (contextual)", bm25_results, bm25_time)

    print_full_chunks("VECTOR (BGE-M3, contextual)", vector_results)
    print_full_chunks("BM25 (contextual)", bm25_results)


if __name__ == "__main__":
    main()
