"""
Recursive (Sentence-Based) chunking, WITH and WITHOUT contextual summaries,
on the 10-lecture evaluation sample: 8, 10, 21, 34, 44, 53, 66, 76, 89, 100.

This is chunk-prep only - no vectors/BM25 indexes yet, that's the next step.
We just need both a "no contextual retrieval" and "with contextual retrieval"
version of the same chunks to build indexes from and compare later. It also
reports the real dollar cost of the contextual-summary step, computed from
the actual token usage the API reported for every call (not an estimate).

Runs lectures concurrently (async, one lecture = one task) since the LLM
summary calls are I/O-bound and independent across lectures. Within a single
lecture, chunks are still processed in order, one call at a time - this
preserves the same prompt-caching behavior we measured in the lecture-46
smoke test (the shared lecture-text prefix stays warm across consecutive
calls for that lecture).

Each lecture's result is saved to its own file in lecture_cache/ AS SOON AS
that lecture finishes, and a lecture already cached is skipped on a re-run.
This matters twice over: a persistent failure on one lecture (rate limits,
etc.) must not throw away the paid-for work already done on the others, and
re-running this script later (e.g. just to regenerate the cost summary, or
because you only care about the chunk data and not the cost) burns no new
tokens at all if every lecture is already cached.

Output: recursive_eval_set_results.json (combined from lecture_cache/*.json)
"""

import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tiktoken
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from openai import AsyncOpenAI, RateLimitError

load_dotenv()
client = AsyncOpenAI()

# Rates per 1,000,000 tokens, verified live against OpenAI's pricing page
# during this project's development.
MODEL_RATES = {
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
}


def calculate_cost(model, input_tokens, output_tokens, cached_tokens=0):
    """input_tokens is the full prompt_tokens value (INCLUDES cached_tokens)."""
    rates = MODEL_RATES[model]
    fresh_input_tokens = input_tokens - cached_tokens
    return (
        fresh_input_tokens / 1_000_000 * rates["input"]
        + cached_tokens / 1_000_000 * rates["cached_input"]
        + output_tokens / 1_000_000 * rates["output"]
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_PATH = REPO_ROOT / "seerah_transcripts.jsonl"
OUTPUT_PATH = REPO_ROOT / "recursive_eval_set_results.json"
CACHE_DIR = REPO_ROOT / "lecture_cache"

TARGET_LECTURES = [8, 10, 21, 34, 44, 53, 66, 76, 89, 100]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
SUMMARY_MODEL = "gpt-5.4-nano"

# Running different lectures concurrently means each one needs its own
# "cold start" cache warm-up (an ~18-25k token full-price read) - caching
# doesn't transfer between different lecture texts. Running too many of
# those cold starts at once is what blew the account's TPM limit last time.
# Keep concurrency low AND stagger new lecture starts so cold starts don't
# pile up on top of each other.
LECTURE_CONCURRENCY_LIMIT = 2
STAGGER_SECONDS = 8
lecture_semaphore = asyncio.Semaphore(LECTURE_CONCURRENCY_LIMIT)

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(TOKENIZER.encode(text))


def count_words(text):
    return len(text.split())


def load_target_lectures():
    lectures = []
    with open(TRANSCRIPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["lecture_number"] in TARGET_LECTURES:
                lectures.append(record)
    found = {r["lecture_number"] for r in lectures}
    missing = set(TARGET_LECTURES) - found
    if missing:
        raise ValueError(f"Lecture(s) not found in dataset: {sorted(missing)}")
    lectures.sort(key=lambda r: TARGET_LECTURES.index(r["lecture_number"]))
    return lectures


async def get_contextual_summary(lecture_text, canonical_title, chunk_text):
    prompt = (
        f"This is a lecture titled \"{canonical_title}\".\n\n"
        f"Full lecture transcript:\n<lecture>\n{lecture_text}\n</lecture>\n\n"
        f"Here is one excerpt taken from that lecture:\n<chunk>\n{chunk_text}\n</chunk>\n\n"
        "In 1-2 short sentences, describe what this specific excerpt covers, "
        "so it can be understood correctly on its own without the rest of the lecture. "
        "Answer with only the summary, nothing else."
    )
    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            break
        except RateLimitError:
            wait_seconds = min(2 ** attempt, 60)
            print(f"    rate limited, retrying in {wait_seconds}s...")
            await asyncio.sleep(wait_seconds)
    else:
        raise RuntimeError("Exceeded max retries on rate limit")

    summary = response.choices[0].message.content.strip()

    usage = response.usage
    cached_tokens = 0
    if usage.prompt_tokens_details is not None:
        cached_tokens = usage.prompt_tokens_details.cached_tokens or 0

    usage_dict = {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cached_tokens": cached_tokens,
    }
    return summary, usage_dict


async def process_lecture(lecture, splitter, stagger_delay):
    cache_path = CACHE_DIR / f"lecture_{lecture['lecture_number']}.json"
    if cache_path.exists():
        print(f"Lecture {lecture['lecture_number']}: already cached, skipping")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    await asyncio.sleep(stagger_delay)
    async with lecture_semaphore:
        result = await _process_lecture_inner(lecture, splitter)

    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


async def _process_lecture_inner(lecture, splitter):
    doc = Document(text=lecture["text"], metadata={
        "lecture_number": lecture["lecture_number"],
        "canonical_title": lecture["canonical_title"],
        "youtube_url": lecture["youtube_url"],
    })
    nodes = splitter.get_nodes_from_documents([doc])
    print(f"Lecture {lecture['lecture_number']} ({lecture['canonical_title']}): {len(nodes)} chunks - starting")

    recursive = []
    recursive_contextual = []

    for i, node in enumerate(nodes):
        chunk_text = node.get_content()

        recursive.append({
            "strategy": "Recursive (Sentence)",
            "chunk_index": i,
            "lecture_number": lecture["lecture_number"],
            "canonical_title": lecture["canonical_title"],
            "youtube_url": lecture["youtube_url"],
            "text": chunk_text,
            "word_count": count_words(chunk_text),
            "token_count": count_tokens(chunk_text),
        })

        summary, usage = await get_contextual_summary(lecture["text"], lecture["canonical_title"], chunk_text)
        combined_text = summary + "\n\n" + chunk_text

        recursive_contextual.append({
            "strategy": "Recursive + Contextual",
            "chunk_index": i,
            "lecture_number": lecture["lecture_number"],
            "canonical_title": lecture["canonical_title"],
            "youtube_url": lecture["youtube_url"],
            "summary": summary,
            "text": combined_text,
            "word_count": count_words(combined_text),
            "token_count": count_tokens(combined_text),
            "usage": usage,
        })

    print(f"Lecture {lecture['lecture_number']}: done ({len(nodes)} chunks)")
    return {"recursive": recursive, "recursive_contextual": recursive_contextual}


def compute_cost_summary(recursive_contextual):
    total_input = sum(c["usage"]["input_tokens"] for c in recursive_contextual)
    total_output = sum(c["usage"]["output_tokens"] for c in recursive_contextual)
    total_cached = sum(c["usage"]["cached_tokens"] for c in recursive_contextual)
    total_cost = calculate_cost(SUMMARY_MODEL, total_input, total_output, total_cached)

    return {
        "model": SUMMARY_MODEL,
        "num_calls": len(recursive_contextual),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cached_tokens": total_cached,
        "cache_hit_rate": round(total_cached / total_input, 4) if total_input else 0,
        "total_cost_usd": round(total_cost, 5),
    }


async def main():
    lectures = load_target_lectures()
    print(f"Loaded {len(lectures)} lectures: {[l['lecture_number'] for l in lectures]}")
    print(f"Concurrency limit: {LECTURE_CONCURRENCY_LIMIT}, staggered {STAGGER_SECONDS}s apart\n")

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    tasks = [
        process_lecture(lecture, splitter, stagger_delay=i * STAGGER_SECONDS)
        for i, lecture in enumerate(lectures)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    recursive = []
    recursive_contextual = []
    failed = []
    for lecture, result in zip(lectures, results):
        if isinstance(result, Exception):
            failed.append(lecture["lecture_number"])
            print(f"Lecture {lecture['lecture_number']} FAILED: {result}")
            continue
        recursive.extend(result["recursive"])
        recursive_contextual.extend(result["recursive_contextual"])

    cost_summary = compute_cost_summary(recursive_contextual)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "Recursive (Sentence)": recursive,
                "Recursive + Contextual": recursive_contextual,
                "cost_summary": cost_summary,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"Contextual summary cost: {json.dumps(cost_summary, indent=2)}")
    if failed:
        print(f"Lectures that failed and were skipped: {failed} - re-run the script to retry just these "
              f"(already-cached lectures are skipped automatically).")


if __name__ == "__main__":
    asyncio.run(main())
