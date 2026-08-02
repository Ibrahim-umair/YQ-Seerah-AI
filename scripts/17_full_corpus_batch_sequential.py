"""
Full-corpus ingestion via OpenAI's Batch API, done properly this time:
sequential waves of whole lectures, each wave sized to stay under the
account's 2,000,000 enqueued-token cap for gpt-5.4-nano batch jobs.

Why waves: a single request here carries the FULL lecture text (~17k tokens
avg) plus one chunk, because the contextual summary needs full-lecture
context. Submitting all ~1,459 pending chunks in one batch meant ~26M
enqueued tokens - 13x over the cap - and got rejected outright before ever
queueing (see conversation: the first attempt at this failed with
token_limit_exceeded). Waves pack whole lectures (using each lecture's real
token count via tiktoken, not an average) into a batch until the next
lecture would push it over a safe budget, then start a new wave.

Each wave: submit -> poll with exponential backoff until completed -> the
lectures in that wave are now fully done -> write their final per-lecture
cache files -> move to the next wave. All waves run in one continuous
script execution (meant to run in the background - this is a long job).

SAME caching protocol as before:
  - full_corpus_cache/lecture_{n}.json = a fully-completed lecture.
  - full_corpus_cache/lecture_{n}.partial.json = chunks already done via the
    earlier interactive run for lectures 26/42/43 - only their REMAINING
    chunks go into a wave; already-done chunks are merged back in.
  - lecture_cache/lecture_{n}.json = the 10 eval-pilot lectures, reused
    directly.

Resume behavior: there's no separate wave-state file - resume works because
every wave finalizes by writing real per-lecture cache files immediately, so
re-running this script re-derives what's still pending from those cache
files directly (already-finished lectures are skipped, waves get rebuilt
from whatever's left). One real gap this doesn't cover: if the process dies
while a wave's batch is mid-flight (submitted, not yet polled to
completion), that in-flight batch's ID isn't remembered - re-running will
submit a fresh batch for the same lectures rather than resuming that one.
Wasteful but not incorrect (the old batch just finishes unobserved); worth
knowing rather than assuming full crash-proofing.

Usage: python scripts/17_full_corpus_batch_sequential.py
       (safe to re-run if interrupted between waves)
"""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tiktoken
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
client = OpenAI()

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_PATH = REPO_ROOT / "seerah_transcripts.jsonl"
OUTPUT_PATH = REPO_ROOT / "full_corpus_chunks.json"
CACHE_DIR = REPO_ROOT / "full_corpus_cache"
EVAL_CACHE_DIR = REPO_ROOT / "lecture_cache"
WAVE_INPUT_DIR = REPO_ROOT / "batch_wave_inputs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
SUMMARY_MODEL = "gpt-5.4-nano"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "seerah_full_corpus_contextual"
EMBED_BATCH_SIZE = 50

WAVE_TOKEN_BUDGET = 1_800_000  # under the 2M cap, with safety margin

POLL_START_SECONDS = 30
POLL_MAX_SECONDS = 600
POLL_BACKOFF_FACTOR = 1.6

MODEL_RATES = {"gpt-5.4-nano": {"input": 0.10, "cached_input": 0.01, "output": 0.625}}  # batch = 50% off standard

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def calculate_cost(model, input_tokens, output_tokens, cached_tokens=0):
    rates = MODEL_RATES[model]
    fresh = input_tokens - cached_tokens
    return fresh / 1e6 * rates["input"] + cached_tokens / 1e6 * rates["cached_input"] + output_tokens / 1e6 * rates["output"]


def count_tokens(text):
    return len(TOKENIZER.encode(text))


def count_words(text):
    return len(text.split())


def load_all_lectures():
    lectures = []
    with open(TRANSCRIPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lectures.append(json.loads(line))
    lectures.sort(key=lambda r: r["lecture_number"])
    return lectures


def try_reuse_eval_cache(lecture_number):
    eval_path = EVAL_CACHE_DIR / f"lecture_{lecture_number}.json"
    if not eval_path.exists():
        return None
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)
    return {"plain": eval_data["recursive"], "contextual": eval_data["recursive_contextual"]}


def build_prompt(lecture_text, canonical_title, chunk_text):
    return (
        f"This is a lecture titled \"{canonical_title}\".\n\n"
        f"Full lecture transcript:\n<lecture>\n{lecture_text}\n</lecture>\n\n"
        f"Here is one excerpt taken from that lecture:\n<chunk>\n{chunk_text}\n</chunk>\n\n"
        "In 1-2 short sentences, describe what this specific excerpt covers, "
        "so it can be understood correctly on its own without the rest of the lecture. "
        "Answer with only the summary, nothing else."
    )


def gather_lecture_units(lectures, splitter):
    """Returns (already_done: {n: {"plain","contextual"}}, units: [{"lecture_number",
    "items": [...], "existing_plain", "existing_contextual", "estimated_tokens"}])
    - one unit per lecture still needing work, its remaining chunks as indivisible items."""
    already_done = {}
    units = []

    for lecture in lectures:
        n = lecture["lecture_number"]
        final_path = CACHE_DIR / f"lecture_{n}.json"
        partial_path = CACHE_DIR / f"lecture_{n}.partial.json"

        if final_path.exists():
            with open(final_path, encoding="utf-8") as f:
                already_done[n] = json.load(f)
            continue

        reused = try_reuse_eval_cache(n)
        if reused is not None:
            CACHE_DIR.mkdir(exist_ok=True)
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(reused, f, ensure_ascii=False, indent=2)
            already_done[n] = reused
            continue

        doc = Document(text=lecture["text"], metadata={})
        nodes = splitter.get_nodes_from_documents([doc])

        existing_plain, existing_contextual = [], []
        start_index = 0
        if partial_path.exists():
            with open(partial_path, encoding="utf-8") as f:
                partial = json.load(f)
            existing_plain, existing_contextual = partial["plain"], partial["contextual"]
            start_index = len(existing_contextual)

        items = []
        for i in range(start_index, len(nodes)):
            chunk_text = nodes[i].get_content()
            prompt = build_prompt(lecture["text"], lecture["canonical_title"], chunk_text)
            items.append({
                "lecture_number": n,
                "chunk_index": i,
                "canonical_title": lecture["canonical_title"],
                "youtube_url": lecture["youtube_url"],
                "chunk_text": chunk_text,
                "prompt": prompt,
                "prompt_tokens": count_tokens(prompt),
            })

        if items:
            units.append({
                "lecture_number": n,
                "items": items,
                "existing_plain": existing_plain,
                "existing_contextual": existing_contextual,
                "estimated_tokens": sum(it["prompt_tokens"] for it in items),
            })

    return already_done, units


def pack_waves(units):
    """Greedy bin-packing: whole lectures per wave, each wave under WAVE_TOKEN_BUDGET."""
    waves = []
    current_wave = []
    current_tokens = 0

    for unit in sorted(units, key=lambda u: -u["estimated_tokens"]):
        if unit["estimated_tokens"] > WAVE_TOKEN_BUDGET:
            # a single lecture alone exceeds the budget (shouldn't happen at ~17k/chunk
            # x even the longest lecture's chunk count, but guard anyway) - gets its own wave
            waves.append([unit])
            continue
        if current_tokens + unit["estimated_tokens"] > WAVE_TOKEN_BUDGET and current_wave:
            waves.append(current_wave)
            current_wave = []
            current_tokens = 0
        current_wave.append(unit)
        current_tokens += unit["estimated_tokens"]

    if current_wave:
        waves.append(current_wave)
    return waves


def submit_wave(wave_units, wave_index):
    WAVE_INPUT_DIR.mkdir(exist_ok=True)
    input_path = WAVE_INPUT_DIR / f"wave_{wave_index}.jsonl"

    total_items = 0
    with open(input_path, "w", encoding="utf-8") as f:
        for unit in wave_units:
            for item in unit["items"]:
                custom_id = f"{item['lecture_number']}_{item['chunk_index']}"
                body = {"model": SUMMARY_MODEL, "messages": [{"role": "user", "content": item["prompt"]}], "temperature": 0}
                f.write(json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}) + "\n")
                total_items += 1

    lecture_numbers = [u["lecture_number"] for u in wave_units]
    est_tokens = sum(u["estimated_tokens"] for u in wave_units)
    print(f"\nWave {wave_index}: {len(wave_units)} lectures {lecture_numbers}, {total_items} chunks, ~{est_tokens:,} tokens")

    uploaded = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"  submitted: {batch.id}")
    return batch.id


def poll_until_done(batch_id):
    wait = POLL_START_SECONDS
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  [{time.strftime('%H:%M:%S')}] status: {batch.status}  counts: {batch.request_counts}")
        if batch.status == "completed":
            return batch
        if batch.status in ("failed", "expired", "cancelled"):
            print(f"  wave batch did not complete: {batch.status}. Errors: {batch.errors}")
            return batch
        time.sleep(wait)
        wait = min(wait * POLL_BACKOFF_FACTOR, POLL_MAX_SECONDS)


def finalize_wave(batch, wave_units):
    output_content = client.files.content(batch.output_file_id).text

    results_by_custom_id = {}
    for line in output_content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        response_body = record["response"]["body"]
        summary = response_body["choices"][0]["message"]["content"].strip()
        usage = response_body["usage"]
        cached = 0
        if usage.get("prompt_tokens_details"):
            cached = usage["prompt_tokens_details"].get("cached_tokens", 0) or 0
        results_by_custom_id[record["custom_id"]] = {
            "summary": summary,
            "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"], "cached_tokens": cached},
        }

    for unit in wave_units:
        n = unit["lecture_number"]
        plain = list(unit["existing_plain"])
        contextual = list(unit["existing_contextual"])

        for item in unit["items"]:
            custom_id = f"{n}_{item['chunk_index']}"
            result = results_by_custom_id.get(custom_id)
            if result is None:
                print(f"  WARNING: no result for {custom_id}")
                continue
            chunk_text = item["chunk_text"]
            plain.append({
                "strategy": "Recursive (Sentence)", "chunk_index": item["chunk_index"], "lecture_number": n,
                "canonical_title": item["canonical_title"], "youtube_url": item["youtube_url"],
                "text": chunk_text, "word_count": count_words(chunk_text), "token_count": count_tokens(chunk_text),
            })
            combined_text = result["summary"] + "\n\n" + chunk_text
            contextual.append({
                "strategy": "Recursive + Contextual", "chunk_index": item["chunk_index"], "lecture_number": n,
                "canonical_title": item["canonical_title"], "youtube_url": item["youtube_url"],
                "summary": result["summary"], "text": combined_text,
                "word_count": count_words(combined_text), "token_count": count_tokens(combined_text),
                "usage": result["usage"],
            })

        final_path = CACHE_DIR / f"lecture_{n}.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump({"plain": plain, "contextual": contextual}, f, ensure_ascii=False, indent=2)
        (CACHE_DIR / f"lecture_{n}.partial.json").unlink(missing_ok=True)
        print(f"  lecture {n}: finalized ({len(contextual)} chunks)")


def build_vector_index(contextual_chunks):
    print(f"\nEmbedding {len(contextual_chunks)} chunks with {EMBEDDING_MODEL}...")
    qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
    if qdrant.collection_exists(COLLECTION_NAME):
        print("  collection already exists, deleting and rebuilding")
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))

    total_tokens = 0
    for batch_start in range(0, len(contextual_chunks), EMBED_BATCH_SIZE):
        batch = contextual_chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        print(f"  embedding batch {batch_start}-{batch_start+len(batch)}...")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        total_tokens += response.usage.total_tokens
        points = [
            PointStruct(id=batch_start + i, vector=response.data[i].embedding, payload={
                "lecture_number": c["lecture_number"], "canonical_title": c["canonical_title"],
                "youtube_url": c["youtube_url"], "chunk_index": c["chunk_index"], "text": c["text"],
            }) for i, c in enumerate(batch)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    embed_cost = total_tokens / 1e6 * 0.13
    print(f"  done - {len(contextual_chunks)} points upserted. Tokens: {total_tokens:,}  Cost: ${embed_cost:.4f}")
    return total_tokens, embed_cost


def main():
    lectures = load_all_lectures()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    already_done, units = gather_lecture_units(lectures, splitter)

    if units:
        waves = pack_waves(units)
        print(f"{len(units)} lectures need processing, packed into {len(waves)} sequential waves")
        for i, wave in enumerate(waves):
            batch_id = submit_wave(wave, i)
            batch = poll_until_done(batch_id)
            if batch.status != "completed":
                print(f"Stopping - wave {i} did not complete successfully. Re-run this script to retry.")
                return
            finalize_wave(batch, wave)
        print("\nAll waves complete.")
    else:
        print("Nothing pending - all lectures already cached.")

    # re-gather to pick up everything now cached, then build the vector index
    already_done, remaining = gather_lecture_units(lectures, splitter)
    if remaining:
        print(f"WARNING: {len(remaining)} lectures still pending after waves - something's off, investigate before embedding.")
        return

    contextual = []
    for n, data in already_done.items():
        contextual.extend(data["contextual"])

    print(f"\nAll {len(already_done)} lectures ready. Total chunks: {len(contextual)}")
    embed_tokens, embed_cost = build_vector_index(contextual)

    output = {
        "Recursive + Contextual": contextual,
        "embedding_cost_summary": {"model": EMBEDDING_MODEL, "total_tokens": embed_tokens, "total_cost_usd": round(embed_cost, 5)},
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
