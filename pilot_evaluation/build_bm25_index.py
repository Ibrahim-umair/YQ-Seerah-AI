"""
Builds local BM25 (keyword) indexes for both chunk variants. No embeddings,
no GPU, no external service - classical lexical search, fully independent
of Qdrant, kept as the "keyword search" side of the vector-vs-BM25
comparison.

pip install llama-index-retrievers-bm25
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

LOCAL_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = LOCAL_DIR / "recursive_eval_set_results.json"
BM25_DIR = LOCAL_DIR / "bm25_indexes"

INDEXES = {
    "Recursive (Sentence)": "recursive_plain",
    "Recursive + Contextual": "recursive_contextual",
}


def build_bm25(name, chunks):
    nodes = [
        TextNode(
            text=c["text"],
            id_=str(i),
            metadata={
                "lecture_number": c["lecture_number"],
                "canonical_title": c["canonical_title"],
                "youtube_url": c["youtube_url"],
                "chunk_index": c["chunk_index"],
            },
        )
        for i, c in enumerate(chunks)
    ]
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

    out_dir = BM25_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever.persist(str(out_dir))
    print(f"Built BM25 index '{name}' from {len(nodes)} chunks -> {out_dir}")


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    for strategy_key, name in INDEXES.items():
        build_bm25(name, data[strategy_key])

    print("\nAll BM25 indexes built.")


if __name__ == "__main__":
    main()
