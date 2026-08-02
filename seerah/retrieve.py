"""Retrieval over the full 104-lecture corpus.

Loads the vector store and the BM25 index once, then answers queries against
either. Both retrievers return the same result shape, so callers (the CLI now,
the web app later) don't need to care which one produced a hit.
"""

import time
from dataclasses import dataclass

from openai import OpenAI
from qdrant_client import QdrantClient
from llama_index.retrievers.bm25 import BM25Retriever

from seerah import config

DEFAULT_TOP_K = 10


@dataclass
class Hit:
    score: float
    lecture_number: int
    canonical_title: str
    youtube_url: str
    chunk_index: int
    text: str

    @property
    def citation(self):
        return f"Lecture {self.lecture_number}: {self.canonical_title} ({self.youtube_url})"


class Retriever:
    """Holds the loaded indexes. Construct once, query many times - loading the
    BM25 index costs a couple of seconds and shouldn't happen per query."""

    def __init__(self, load_bm25=True, load_vector=True):
        self.openai = OpenAI() if load_vector else None
        self.qdrant = None
        self.bm25 = None

        if load_vector:
            self.qdrant = QdrantClient(url=config.QDRANT_URL, timeout=30)
            if not self.qdrant.collection_exists(config.COLLECTION_NAME):
                raise SystemExit(
                    f"Qdrant collection '{config.COLLECTION_NAME}' not found.\n"
                    f"Start Qdrant with `docker compose up -d`, then run "
                    f"`python -m seerah.ingest.embed`."
                )

        if load_bm25:
            if not config.BM25_DIR.exists() or not any(config.BM25_DIR.iterdir()):
                raise SystemExit(
                    f"BM25 index not found at {config.BM25_DIR}.\n"
                    f"Build it with `python -m seerah.ingest.bm25`."
                )
            self.bm25 = BM25Retriever.from_persist_dir(str(config.BM25_DIR))

    def embed_query(self, query):
        response = self.openai.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
        return response.data[0].embedding

    def vector_search(self, query, top_k=DEFAULT_TOP_K):
        """Returns (hits, embed_seconds, search_seconds)."""
        t0 = time.perf_counter()
        query_vector = self.embed_query(query)
        embed_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        response = self.qdrant.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        search_seconds = time.perf_counter() - t0

        hits = [
            Hit(
                score=p.score,
                lecture_number=p.payload["lecture_number"],
                canonical_title=p.payload["canonical_title"],
                youtube_url=p.payload.get("youtube_url", ""),
                chunk_index=p.payload["chunk_index"],
                text=p.payload["text"],
            )
            for p in response.points
        ]
        return hits, embed_seconds, search_seconds

    def bm25_search(self, query, top_k=DEFAULT_TOP_K):
        """Returns (hits, search_seconds)."""
        self.bm25.similarity_top_k = top_k
        t0 = time.perf_counter()
        nodes = self.bm25.retrieve(query)[:top_k]
        search_seconds = time.perf_counter() - t0

        hits = [
            Hit(
                score=n.score,
                lecture_number=n.metadata["lecture_number"],
                canonical_title=n.metadata["canonical_title"],
                youtube_url=n.metadata.get("youtube_url", ""),
                chunk_index=n.metadata["chunk_index"],
                text=n.get_content(),
            )
            for n in nodes
        ]
        return hits, search_seconds
