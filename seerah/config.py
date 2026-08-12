"""Shared paths, model choices and constants for the whole pipeline.

Everything that more than one stage needs to agree on lives here, so the
stages can't quietly drift apart from each other the way the earlier
standalone scripts did.
"""

import os
import sys
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# --- Stage inputs / outputs -------------------------------------------------
TRANSCRIPTS_PATH = DATA_DIR / "seerah_transcripts.jsonl"
MANIFEST_PATH = DATA_DIR / "chunking_manifest.json"
PLAIN_CHUNKS_PATH = DATA_DIR / "chunks_plain.json"
CONTEXTUAL_CHUNKS_PATH = DATA_DIR / "chunks_contextual.json"

CONTEXTUAL_CACHE_DIR = DATA_DIR / "contextual_cache"
BATCH_INPUT_DIR = DATA_DIR / "batch_wave_inputs"
BM25_DIR = DATA_DIR / "bm25_index"

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# --- Models -----------------------------------------------------------------
SUMMARY_MODEL = "gpt-5.4-nano"
ANSWER_MODEL = "gpt-5.6-luna"
JUDGE_MODEL = "gpt-5.6-luna"  # same tier as ANSWER_MODEL for now - named separately so the
                              # judge can be swapped independently later without touching generation
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# --- Vector store -----------------------------------------------------------
# localhost is correct when running directly on the host (bot.py, cli.py, or
# the API run outside Docker) - override to http://qdrant:6333 when the API
# itself runs inside the docker-compose network, since containers reach each
# other by service name, not localhost (same reasoning as POSTGRES_HOST below).
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "seerah_full_corpus_contextual"
EMBED_BATCH_SIZE = 50

# --- Conversation/feedback log (runs via docker-compose.yml) ----------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "seerah")
POSTGRES_USER = os.getenv("POSTGRES_USER", "seerah")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "seerah")

# --- Hybrid search (Reciprocal Rank Fusion) ----------------------------------
# RRF combines vector + BM25 by RANK POSITION, not raw score - the two are on
# incomparable scales (cosine ~0.5-0.6 vs BM25 ~5-9), so any score-weighted
# fusion would just be dominated by whichever retriever's numbers are bigger.
# k=60 is the standard constant from the original RRF paper, but sweeping k
# against this project's 304-question eval set (seerah.eval.sweep_rrf_k) showed
# k=10 winning on full-coverage@10 across every value tested (1-60) while
# staying within 0.004 of the best recall@10 - see the README for the numbers.
RRF_K = 10
# Fetch this many results from EACH retriever before fusing, then trim to the
# caller's requested top_k. Must exceed top_k, or a chunk that ranks just
# outside one retriever's shallow top-k never gets the chance to be pulled up
# by a strong ranking from the other retriever.
RRF_CANDIDATE_POOL = 50

# --- Agent (agentic RAG: the LLM decides when/how many times to search) -----
AGENT_MAX_ITERATIONS = 6    # hard cap on search rounds; the model is forced to
                            # answer (tools withheld) if it still wants to search past this.
                            # Raised from 3 - each iteration resends the full accumulated
                            # conversation, so cost grows faster than linearly with this value.
AGENT_SEARCH_TOP_K = 5      # chunks per agent search - smaller than the plain top_k=10
                            # default since results accumulate in the conversation across iterations

# --- Batch API --------------------------------------------------------------
# A single contextual-summary request carries the FULL lecture (~17k tokens) as
# context, so a naive "submit everything" batch lands at ~26M enqueued tokens -
# 13x over the account's 2M cap. Work is packed into waves under this budget.
WAVE_TOKEN_BUDGET = 1_800_000
POLL_START_SECONDS = 30
POLL_MAX_SECONDS = 600
POLL_BACKOFF_FACTOR = 1.6

# Batch API pricing = 50% off standard, per 1M tokens.
MODEL_RATES = {
    "gpt-5.4-nano": {"input": 0.10, "cached_input": 0.01, "output": 0.625},
    # Short-context tier only - this project's requests (a chunk is ~800 tokens,
    # even a full 6-iteration agentic loop's accumulated history) never approach
    # the long-context threshold. "Cache writes" pricing isn't tracked here since
    # the OpenAI usage object doesn't expose a cache-write token count to charge it against.
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
}
EMBEDDING_RATE = 0.13  # text-embedding-3-large, per 1M tokens

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(TOKENIZER.encode(text))


def count_words(text):
    return len(text.split())


def summary_cost(model, input_tokens, output_tokens, cached_tokens=0):
    rates = MODEL_RATES[model]
    fresh = input_tokens - cached_tokens
    return (
        fresh / 1e6 * rates["input"]
        + cached_tokens / 1e6 * rates["cached_input"]
        + output_tokens / 1e6 * rates["output"]
    )


def use_utf8_stdout():
    """Windows consoles default to cp1252 and blow up on Arabic script."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
