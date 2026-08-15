"""Stage 3 - embed the contextual chunks and load them into Qdrant.

    Input:  data/chunks_contextual.json
    Output: Qdrant collection `seerah_full_corpus_contextual` (3072-dim, cosine)

Cheap: ~$0.28 and a few minutes for the whole 2,763-chunk corpus. This is the
one stage a reviewer cloning the repo actually has to run, because the vector
store itself is too large to commit - everything upstream of it is already in
the repository as a committed artifact.

If data/chunks_contextual_with_timestamps.json is present (same chunks, plus
each one's transcript sentences individually timestamped), each point's
payload also gets a 'sentences' field - used only by SeerahAgent's citation
refinement pass to report a precise in-lecture moment instead of just the
chunk's own start. Embedding itself is unaffected either way: it's still
computed from 'text' alone, so this file's presence never changes what gets
embedded or costs anything extra. Missing that file degrades gracefully -
citation refinement just falls back to chunk-start timestamps.

Requires Qdrant to be running:  docker compose up -d

Embedding model is OpenAI text-embedding-3-large. It beat a local BGE-M3 setup
on every retrieval metric during evaluation (see the README), at the cost of
needing a live API call per query instead of running offline.

Usage:
    python -m seerah.ingest.embed           # skips if the collection is already complete
    python -m seerah.ingest.embed --force   # delete and rebuild the collection
    python -m seerah.ingest.embed --verify  # check the live collection against the artifact
"""

import argparse

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from seerah import artifacts, config

client = OpenAI()


def connect():
    try:
        qdrant = QdrantClient(url=config.QDRANT_URL, timeout=60)
        qdrant.get_collections()
        return qdrant
    except Exception as exc:
        raise SystemExit(
            f"Could not reach Qdrant at {config.QDRANT_URL}: {exc}\n"
            f"Start it with:  docker compose up -d"
        )


def verify(qdrant, chunks):
    """Checks the live collection actually matches the artifact.

    Point count alone is not enough, on two counts. An ungraceful Docker
    shutdown once left this collection with intact vectors and 44% of payloads
    gone, so payload presence is checked. And re-running stage 2 can change a
    chunk's text without changing how many there are - repairing lectures
    26/42/43 rewrote 92 chunks and left the count at 2,763 - so the stored text
    is compared against the artifact too. Otherwise a stale collection reports
    itself healthy and quietly serves outdated content."""
    if not qdrant.collection_exists(config.COLLECTION_NAME):
        print(f"collection '{config.COLLECTION_NAME}' does not exist")
        return False

    count = qdrant.get_collection(config.COLLECTION_NAME).points_count
    if count != len(chunks):
        print(f"collection has {count} points, artifact has {len(chunks)} chunks")
        return False

    missing, stale, lectures, offset = 0, 0, set(), None
    while True:
        points, offset = qdrant.scroll(
            collection_name=config.COLLECTION_NAME, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            if not payload.get("text") or "lecture_number" not in payload or "chunk_index" not in payload:
                missing += 1
                continue
            lectures.add(payload["lecture_number"])
            # points are upserted with id == index into the artifact
            if not isinstance(p.id, int) or p.id >= len(chunks) or payload["text"] != chunks[p.id]["text"]:
                stale += 1
        if offset is None:
            break

    if missing:
        print(f"collection has {missing} points with missing/empty payloads")
    if stale:
        print(f"collection has {stale} points whose text differs from {config.CONTEXTUAL_CHUNKS_PATH.name} "
              f"- it is out of date and needs re-embedding")
    if missing or stale:
        return False

    print(f"VERIFIED: {count} points, {len(lectures)} lectures, 0 bad payloads, text matches the artifact")
    return True


def build(qdrant, chunks, sentence_lookup):
    if qdrant.collection_exists(config.COLLECTION_NAME):
        print(f"  collection '{config.COLLECTION_NAME}' exists - deleting and rebuilding")
        qdrant.delete_collection(config.COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
    )

    total_tokens = 0
    for start in range(0, len(chunks), config.EMBED_BATCH_SIZE):
        batch = chunks[start:start + config.EMBED_BATCH_SIZE]
        print(f"  embedding {start}-{start + len(batch)} of {len(chunks)}...")
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL, input=[c["text"] for c in batch]
        )
        total_tokens += response.usage.total_tokens
        qdrant.upsert(
            collection_name=config.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=start + i,
                    vector=response.data[i].embedding,
                    payload={
                        "lecture_number": c["lecture_number"],
                        "canonical_title": c["canonical_title"],
                        "youtube_url": c["youtube_url"],
                        "chunk_index": c["chunk_index"],
                        "text": c["text"],
                        "start_timestamp": c.get("start_timestamp", ""),
                        "start_timestamp_seconds": c.get("start_timestamp_seconds", 0.0),
                        "sentences": sentence_lookup.get((c["lecture_number"], c["chunk_index"]), []),
                    },
                )
                for i, c in enumerate(batch)
            ],
        )

    cost = total_tokens / 1e6 * config.EMBEDDING_RATE
    print(f"  upserted {len(chunks)} points. Tokens: {total_tokens:,}  Cost: ${cost:.4f}")
    return total_tokens, cost


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="delete and rebuild even if complete")
    parser.add_argument("--verify", action="store_true", help="check the live collection, change nothing")
    args = parser.parse_args()

    chunks = artifacts.read_chunks(config.CONTEXTUAL_CHUNKS_PATH)
    qdrant = connect()

    if args.verify:
        raise SystemExit(0 if verify(qdrant, chunks) else 1)

    if not args.force and verify(qdrant, chunks):
        print("Collection is already complete and healthy - nothing to do. Pass --force to rebuild.")
        return

    sentence_lookup = artifacts.load_sentence_timestamps(config.CONTEXTUAL_CHUNKS_WITH_TIMESTAMPS_PATH)
    if sentence_lookup:
        print(f"  found {config.CONTEXTUAL_CHUNKS_WITH_TIMESTAMPS_PATH.name} - "
              f"embedding with sentence-level timestamps included")
    else:
        print(f"  {config.CONTEXTUAL_CHUNKS_WITH_TIMESTAMPS_PATH.name} not found - "
              f"embedding without sentence-level timestamps (citation refinement "
              f"will fall back to chunk-start timestamps only)")

    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL}...")
    build(qdrant, chunks, sentence_lookup)
    verify(qdrant, chunks)


if __name__ == "__main__":
    main()
