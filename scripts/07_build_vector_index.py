"""
Builds a local Qdrant vector index using BGE-M3 embeddings (via
sentence-transformers, runs on CPU - no external API, no GPU required for
this small a corpus).

Two collections, one per chunk variant, so "with vs without contextual
retrieval" stays a clean comparison:
  - recursive_plain       <- "Recursive (Sentence)" chunks
  - recursive_contextual  <- "Recursive + Contextual" chunks

Requires Qdrant already running locally (see docker-compose.yml):
    docker compose up -d
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "recursive_eval_set_results.json"

QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # BGE-M3's dense embedding size

COLLECTIONS = {
    "Recursive (Sentence)": "recursive_plain",
    "Recursive + Contextual": "recursive_contextual",
}


def build_collection(client, model, collection_name, chunks):
    print(f"\nBuilding collection '{collection_name}' from {len(chunks)} chunks...")

    if client.collection_exists(collection_name):
        print("  collection already exists, deleting and rebuilding")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    texts = [c["text"] for c in chunks]
    print(f"  encoding {len(texts)} chunks with {EMBEDDING_MODEL} (CPU - can take a few minutes)...")
    # BGE-M3 recommends normalized embeddings + cosine similarity for retrieval.
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "lecture_number": c["lecture_number"],
                "canonical_title": c["canonical_title"],
                "youtube_url": c["youtube_url"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name=collection_name, points=points)
    print(f"  done - {len(points)} points upserted into '{collection_name}'")


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading {EMBEDDING_MODEL} locally (downloads once, ~2.2GB, then cached)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = QdrantClient(url=QDRANT_URL)

    for strategy_key, collection_name in COLLECTIONS.items():
        build_collection(client, model, collection_name, data[strategy_key])

    print("\nAll collections built. Check http://localhost:6333/dashboard to inspect them.")


if __name__ == "__main__":
    main()
