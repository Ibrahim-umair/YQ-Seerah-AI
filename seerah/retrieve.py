"""Retrieval over the full 104-lecture corpus.

Loads the vector store and the BM25 index once, then answers queries against
either. Both retrievers return the same result shape, so callers (the CLI now,
the web app later) don't need to care which one produced a hit.
"""

import json
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
    start_timestamp: str = ""
    start_timestamp_seconds: float = 0.0
    sentences: list = None

    @property
    def citation(self):
        return f"Lecture {self.lecture_number}: {self.canonical_title} ({self.youtube_url})"

    @property
    def timestamped_url(self):
        """youtube_url with &t=<seconds>s appended, so a citation jumps straight
        to the moment in the lecture this chunk covers, not just the video."""
        if not self.youtube_url:
            return self.youtube_url
        sep = "&" if "?" in self.youtube_url else "?"
        return f"{self.youtube_url}{sep}t={int(self.start_timestamp_seconds)}s"


class Retriever:
    """Holds the loaded indexes. Construct once, query many times - loading the
    BM25 index costs a couple of seconds and shouldn't happen per query."""

    def __init__(self, load_bm25=True, load_vector=True, load_chunk_lookup=True):
        self.openai = OpenAI() if load_vector else None
        self.qdrant = None
        self.bm25 = None
        self._chunk_lookup = None

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

        if load_chunk_lookup:
            path = config.CONTEXTUAL_CHUNKS_WITH_TIMESTAMPS_PATH
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    chunks = json.load(f)["chunks"]
                self._chunk_lookup = {(c["lecture_number"], c["chunk_index"]): c for c in chunks}

    def get_predecessor(self, hit):
        """The chunk immediately before `hit` in the same lecture (chunk_index
        - 1), as a Hit - used only to look slightly earlier than a retrieved
        chunk's own start when refining a citation's timestamp, since a chunk
        can open mid-story even though it never opens mid-sentence (chunking
        respects sentence boundaries, not narrative ones).

        Returns None if `hit` is a lecture's first chunk, or if the
        with-timestamps chunk file isn't available (load_chunk_lookup=False,
        or the file is missing - callers should treat that as "no predecessor
        available" rather than an error, same as chunk_index == 0)."""
        if self._chunk_lookup is None or hit.chunk_index == 0:
            return None
        chunk = self._chunk_lookup.get((hit.lecture_number, hit.chunk_index - 1))
        if chunk is None:
            return None
        return Hit(
            score=0.0,
            lecture_number=chunk["lecture_number"],
            canonical_title=chunk["canonical_title"],
            youtube_url=chunk["youtube_url"],
            chunk_index=chunk["chunk_index"],
            text=chunk["text"],
            start_timestamp=chunk.get("start_timestamp", ""),
            start_timestamp_seconds=chunk.get("start_timestamp_seconds", 0.0),
            sentences=chunk.get("sentences") or [],
        )

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
                start_timestamp=p.payload.get("start_timestamp", ""),
                start_timestamp_seconds=p.payload.get("start_timestamp_seconds", 0.0),
                sentences=p.payload.get("sentences") or [],
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
                start_timestamp=n.metadata.get("start_timestamp", ""),
                start_timestamp_seconds=n.metadata.get("start_timestamp_seconds", 0.0),
                sentences=n.metadata.get("sentences") or [],
            )
            for n in nodes
        ]
        return hits, search_seconds

    def hybrid_search(self, query, top_k=DEFAULT_TOP_K, candidate_pool=config.RRF_CANDIDATE_POOL, k=None):
        """Fuses vector + BM25 via Reciprocal Rank Fusion. Returns (hits, timings).

        Each retriever is queried at `candidate_pool` depth (wider than top_k),
        so a chunk that just misses one retriever's shallow top-k still gets a
        chance to be pulled up by a strong ranking from the other.

        `k` overrides config.RRF_K for this call - useful for sweeping the
        constant without touching global config (see seerah.eval.sweep_rrf_k).
        """
        vector_hits, embed_seconds, vector_seconds = self.vector_search(query, candidate_pool)
        bm25_hits, bm25_seconds = self.bm25_search(query, candidate_pool)

        t0 = time.perf_counter()
        fused_hits = reciprocal_rank_fusion((vector_hits, bm25_hits), config.RRF_K if k is None else k, top_k)
        fuse_seconds = time.perf_counter() - t0

        timings = {
            "embed_seconds": embed_seconds,
            "vector_seconds": vector_seconds,
            "bm25_seconds": bm25_seconds,
            "fuse_seconds": fuse_seconds,
        }
        return fused_hits, timings


def reciprocal_rank_fusion(result_lists, k, top_k):
    """Fuses any number of already-ranked Hit lists into one, by rank position
    alone (never raw score - the lists may be on incomparable scales).

    Pulled out as a standalone function, separate from hybrid_search's network
    calls, so a k-value sweep can fetch each retriever's candidates ONCE and
    re-fuse them at many k values purely in memory - no repeated embedding
    calls or index queries just to test a different k.

    Chunks are deduplicated by (lecture_number, chunk_index) - the same chunk
    appearing in more than one list is fused once, not double-counted.
    """
    rrf_scores = {}
    representative = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            key = (hit.lecture_number, hit.chunk_index)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            representative.setdefault(key, hit)

    ranked_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    return [
        Hit(
            score=rrf_scores[key],
            lecture_number=representative[key].lecture_number,
            canonical_title=representative[key].canonical_title,
            youtube_url=representative[key].youtube_url,
            chunk_index=representative[key].chunk_index,
            text=representative[key].text,
            start_timestamp=representative[key].start_timestamp,
            start_timestamp_seconds=representative[key].start_timestamp_seconds,
            sentences=representative[key].sentences,
        )
        for key in ranked_keys
    ]
