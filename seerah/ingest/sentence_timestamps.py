"""Stage 3b - sync sentence-level timestamps onto an ALREADY-BUILT Qdrant
collection and BM25 index, without re-embedding.

    Input:  data/chunks_contextual_with_timestamps.json (if present)
    Output: a 'sentences' payload field on every Qdrant point + BM25 node

embed.py and bm25.py already include 'sentences' automatically on a FRESH
build if the with-timestamps file exists (see their own docstrings). This
script is the other half: a way to add or remove that enrichment on an
EXISTING collection in place - no re-embedding, no cost, reversible in
either direction independent of which code version you're running.

Two directions:
    python -m seerah.ingest.sentence_timestamps          # sync - add real
        sentences if chunks_contextual_with_timestamps.json exists, clear
        them if it doesn't
    python -m seerah.ingest.sentence_timestamps --clear   # force-clear even
        if that file exists - e.g. to reproduce exactly what a commit from
        before this feature existed would have embedded, without needing to
        actually check out old code or touch data/chunks_contextual.json

Never touches vectors or the 'text' payload either way - both directions
are payload-only and safe to run at any time, on any code version, since
old code (that has never heard of 'sentences') simply ignores payload keys
it doesn't ask for.
"""

import argparse
import shutil

from qdrant_client import QdrantClient

from seerah import artifacts, config
from seerah.ingest import bm25 as bm25_stage


def sync_qdrant(qdrant, chunks, sentence_lookup):
    """sentence_lookup empty means clear the field from every point instead
    of setting real values. Points are addressed by id == index into
    chunks (same scheme embed.py upserts with), so this assumes the
    collection was built from this exact chunk file/order."""
    if not sentence_lookup:
        qdrant.delete_payload(
            collection_name=config.COLLECTION_NAME,
            keys=["sentences"],
            points=list(range(len(chunks))),
        )
        return "cleared"

    for i, c in enumerate(chunks):
        key = (c["lecture_number"], c["chunk_index"])
        qdrant.set_payload(
            collection_name=config.COLLECTION_NAME,
            payload={"sentences": sentence_lookup.get(key, [])},
            points=[i],
        )
    return "synced"


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clear", action="store_true",
                        help="remove 'sentences' from every point, even if the with-timestamps file exists")
    args = parser.parse_args()

    qdrant = QdrantClient(url=config.QDRANT_URL, timeout=60)
    if not qdrant.collection_exists(config.COLLECTION_NAME):
        raise SystemExit(
            f"Collection '{config.COLLECTION_NAME}' doesn't exist.\n"
            f"Run `python -m seerah.ingest.embed` first."
        )

    chunks = artifacts.read_chunks(config.CONTEXTUAL_CHUNKS_PATH)
    sentence_lookup = {} if args.clear else artifacts.load_sentence_timestamps(
        config.CONTEXTUAL_CHUNKS_WITH_TIMESTAMPS_PATH)

    action = "Clearing" if not sentence_lookup else "Syncing"
    print(f"{action} 'sentences' on {len(chunks)} Qdrant points...")
    result = sync_qdrant(qdrant, chunks, sentence_lookup)
    print(f"  Qdrant {result}.")

    if config.BM25_DIR.exists():
        shutil.rmtree(config.BM25_DIR)
    count = bm25_stage.build(chunks, sentence_lookup)
    print(f"  BM25 rebuilt to match -> {config.BM25_DIR} ({count} chunks)")


if __name__ == "__main__":
    main()
