"""Stage 4 - build the BM25 keyword index over the same contextual chunks.

    Input:  data/chunks_contextual.json
    Output: data/bm25_index/

Free and local, takes seconds. BM25 is the keyword-search half of the
comparison run in the retrieval evaluation, and the basis for hybrid search.
It indexes exactly the same text the vector store embeds - the contextual
summary prepended to the chunk - so the two retrievers are compared on equal
footing rather than on different inputs.

If data/chunks_contextual_with_timestamps.json is present, each node's
metadata also gets a 'sentences' field (see seerah.ingest.embed for why) -
purely additional metadata, never part of what gets scored. Missing that
file degrades gracefully - citation refinement just falls back to
chunk-start timestamps for any hit BM25 contributes.

Usage:
    python -m seerah.ingest.bm25           # skips if the index already exists
    python -m seerah.ingest.bm25 --force   # rebuild it
"""

import argparse
import shutil

from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from seerah import artifacts, config


def build(chunks, sentence_lookup):
    nodes = [
        TextNode(
            text=c["text"],
            id_=str(i),
            metadata={
                "lecture_number": c["lecture_number"],
                "canonical_title": c["canonical_title"],
                "youtube_url": c["youtube_url"],
                "chunk_index": c["chunk_index"],
                "start_timestamp": c.get("start_timestamp", ""),
                "start_timestamp_seconds": c.get("start_timestamp_seconds", 0.0),
                "sentences": sentence_lookup.get((c["lecture_number"], c["chunk_index"]), []),
            },
        )
        for i, c in enumerate(chunks)
    ]
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    config.BM25_DIR.mkdir(parents=True, exist_ok=True)
    retriever.persist(str(config.BM25_DIR))
    return len(nodes)


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="rebuild even if the index exists")
    args = parser.parse_args()

    chunks = artifacts.read_chunks(config.CONTEXTUAL_CHUNKS_PATH)

    if config.BM25_DIR.exists() and any(config.BM25_DIR.iterdir()):
        if not args.force:
            print(f"{config.BM25_DIR} already exists - using it as is. Pass --force to rebuild.")
            return
        shutil.rmtree(config.BM25_DIR)

    sentence_lookup = artifacts.load_sentence_timestamps(config.CONTEXTUAL_CHUNKS_WITH_TIMESTAMPS_PATH)
    if sentence_lookup:
        print(f"  found {config.CONTEXTUAL_CHUNKS_WITH_TIMESTAMPS_PATH.name} - "
              f"indexing with sentence-level timestamps included")
    else:
        print(f"  {config.CONTEXTUAL_CHUNKS_WITH_TIMESTAMPS_PATH.name} not found - "
              f"indexing without sentence-level timestamps")

    print(f"Building BM25 index from {len(chunks)} contextual chunks...")
    count = build(chunks, sentence_lookup)
    print(f"Indexed {count} chunks -> {config.BM25_DIR}")


if __name__ == "__main__":
    main()
