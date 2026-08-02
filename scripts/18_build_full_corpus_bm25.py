"""
Builds the BM25 (keyword) index for the full 104-lecture corpus, so the
interactive query tool can compare vector (OpenAI large) vs BM25 at full
scale, not just on the 10-lecture pilot.

Output: full_corpus_bm25_index/
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "full_corpus_chunks.json"
BM25_DIR = REPO_ROOT / "full_corpus_bm25_index"


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)["Recursive + Contextual"]

    print(f"Building BM25 index from {len(chunks)} full-corpus chunks...")
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

    BM25_DIR.mkdir(exist_ok=True)
    retriever.persist(str(BM25_DIR))
    print(f"Built BM25 index from {len(nodes)} chunks -> {BM25_DIR}")


if __name__ == "__main__":
    main()
