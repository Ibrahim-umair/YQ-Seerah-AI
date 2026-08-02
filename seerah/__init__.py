"""Seerah Lookup & Summarizer - a RAG application over Yasir Qadhi's 104-part
Seerah lecture series.

Ingestion is split into four independently runnable stages under
`seerah.ingest`; each one reads the previous stage's committed artifact, so any
stage can be re-run on its own without repeating the work before it:

    python -m seerah.ingest.chunk          # transcripts  -> plain chunks
    python -m seerah.ingest.contextualize  # plain chunks -> contextual chunks  ($)
    python -m seerah.ingest.embed          # contextual   -> Qdrant collection  ($)
    python -m seerah.ingest.bm25           # contextual   -> BM25 keyword index

Every stage is a no-op if its output already exists; pass --force to rebuild.
Query the result with:

    python -m seerah.cli
"""

__version__ = "0.1.0"
