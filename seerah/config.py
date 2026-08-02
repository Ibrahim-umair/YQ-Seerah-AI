"""Shared paths, model choices and constants for the whole pipeline.

Everything that more than one stage needs to agree on lives here, so the
stages can't quietly drift apart from each other the way the earlier
standalone scripts did.
"""

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
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# --- Vector store -----------------------------------------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "seerah_full_corpus_contextual"
EMBED_BATCH_SIZE = 50

# --- Batch API --------------------------------------------------------------
# A single contextual-summary request carries the FULL lecture (~17k tokens) as
# context, so a naive "submit everything" batch lands at ~26M enqueued tokens -
# 13x over the account's 2M cap. Work is packed into waves under this budget.
WAVE_TOKEN_BUDGET = 1_800_000
POLL_START_SECONDS = 30
POLL_MAX_SECONDS = 600
POLL_BACKOFF_FACTOR = 1.6

# Batch API pricing = 50% off standard, per 1M tokens.
MODEL_RATES = {"gpt-5.4-nano": {"input": 0.10, "cached_input": 0.01, "output": 0.625}}
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
