"""The four ingestion stages, each runnable on its own.

    python -m seerah.ingest.chunk          # transcripts  -> data/chunks_plain.json
    python -m seerah.ingest.contextualize  # plain chunks -> data/chunks_contextual.json
    python -m seerah.ingest.embed          # contextual   -> Qdrant collection
    python -m seerah.ingest.bm25           # contextual   -> data/bm25_index/

Stages 2 and 3 cost money (OpenAI). Stages 1 and 4 are free and local.
Every stage skips its work when the output already exists, so cloning the repo
and running only the stages you need is the normal path - see the README.
"""
