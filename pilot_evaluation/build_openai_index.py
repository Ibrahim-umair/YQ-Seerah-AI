"""
Builds OpenAI text-embedding-3-large vector collections for BOTH chunk
variants (plain and contextual), so the ongoing evaluation matrix is
BM25 x {plain, contextual} and OpenAI-large x {plain, contextual} - see the
README for why BGE-M3 was dropped from further active testing (kept as
historical evidence, not deleted).

Safe to run at the same time as a local BGE-M3 job: this makes remote network
calls to OpenAI's API, it doesn't compete for the same CPU threads a local
model inference does.

Batches chunks into groups per API call (OpenAI's embeddings endpoint accepts
a list of inputs per request) rather than one call per chunk.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
client = OpenAI()

LOCAL_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = LOCAL_DIR / "recursive_eval_set_results.json"

QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
BATCH_SIZE = 50

COLLECTIONS = {
    "Recursive (Sentence)": "recursive_plain_openai_large",
    "Recursive + Contextual": "recursive_contextual_openai_large",
}


def build_collection(qdrant, collection_name, chunks):
    print(f"\nBuilding '{collection_name}' from {len(chunks)} chunks with {EMBEDDING_MODEL}...")

    if qdrant.collection_exists(collection_name):
        print("  collection already exists, deleting and rebuilding")
        qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    total_tokens = 0
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        print(f"  batch {batch_start}-{batch_start+len(batch)}...")

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        total_tokens += response.usage.total_tokens

        points = [
            PointStruct(
                id=batch_start + i,
                vector=response.data[i].embedding,
                payload={
                    "lecture_number": c["lecture_number"],
                    "canonical_title": c["canonical_title"],
                    "youtube_url": c["youtube_url"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                },
            )
            for i, c in enumerate(batch)
        ]
        qdrant.upsert(collection_name=collection_name, points=points)

    cost = total_tokens / 1_000_000 * 0.13
    print(f"  done - {len(chunks)} points upserted. Tokens: {total_tokens:,}  Cost: ${cost:.4f}")

    scanned, _ = qdrant.scroll(collection_name=collection_name, limit=300, with_payload=True)
    empty = sum(1 for p in scanned if not p.payload)
    print(f"  verification: {len(scanned)} points scanned, {empty} with empty payload")
    return total_tokens


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    qdrant = QdrantClient(url=QDRANT_URL, timeout=60)

    total_tokens = 0
    for strategy_key, collection_name in COLLECTIONS.items():
        total_tokens += build_collection(qdrant, collection_name, data[strategy_key])

    total_cost = total_tokens / 1_000_000 * 0.13
    print(f"\nAll collections built. Total tokens: {total_tokens:,}  Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
