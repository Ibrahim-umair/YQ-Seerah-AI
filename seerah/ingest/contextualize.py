"""Stage 2 - give every chunk a short LLM-written summary of what it covers,
prepended to the chunk before it gets embedded or indexed.

This is Anthropic's "Contextual Retrieval" technique. A chunk read in isolation
loses its context: pronouns lose their referent, "this battle" could be any
battle, and nothing says which lecture it came from. So for each chunk an LLM
reads the ENTIRE lecture plus that one chunk, and writes 1-2 sentences situating
it. On this corpus that measurably improved retrieval - see the README's
evaluation tables.

    Input:  data/chunks_plain.json, data/seerah_transcripts.jsonl
    Output: data/chunks_contextual.json  (+ per-lecture cache, gitignored)

COSTS MONEY. Full corpus from scratch is ~$1.47 and several hours. The
committed artifact means you should never need to do that - see --dry-run.

Why the OpenAI Batch API rather than plain calls: each request carries the full
~17k-token lecture as context, so the synchronous path spends its whole life
fighting the account's 200k TPM limit. Batch uses a separate quota and is 50%
cheaper. But it caps *enqueued* tokens at 2M, and the whole corpus is ~26M, so
work is bin-packed into sequential waves of whole lectures, each under budget,
submitted one at a time.

Resume safety: each lecture's cache records a fingerprint of the exact chunk
texts it was built from. If the chunking changes, the fingerprint stops
matching and that lecture is re-summarized in full. The previous version of
this pipeline keyed resume on chunk *index* instead, which silently spliced
chunks cut two different ways together and lost 2,201 characters of transcript
across lectures 26, 42 and 43. Fingerprinting is what makes that impossible.

Usage:
    python -m seerah.ingest.contextualize --dry-run   # what would run, and what it costs
    python -m seerah.ingest.contextualize             # do the pending work, reuse the rest
    python -m seerah.ingest.contextualize --force     # discard all cache and redo everything
"""

import argparse
import hashlib
import json
import time

from openai import OpenAI

from seerah import artifacts, config

STRATEGY = "Recursive + Contextual"

client = OpenAI()


def fingerprint(lecture_chunks):
    """Identifies an exact chunking of one lecture. Any boundary change - a
    different splitter setting, a repaired transcript - changes this, which is
    what forces a stale lecture to be re-summarized rather than spliced."""
    h = hashlib.sha256()
    for c in lecture_chunks:
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def build_prompt(lecture_text, canonical_title, chunk_text):
    return (
        f"This is a lecture titled \"{canonical_title}\".\n\n"
        f"Full lecture transcript:\n<lecture>\n{lecture_text}\n</lecture>\n\n"
        f"Here is one excerpt taken from that lecture:\n<chunk>\n{chunk_text}\n</chunk>\n\n"
        "In 1-2 short sentences, describe what this specific excerpt covers, "
        "so it can be understood correctly on its own without the rest of the lecture. "
        "Answer with only the summary, nothing else."
    )


# --- cache ------------------------------------------------------------------

def cache_path(lecture_number):
    return config.CONTEXTUAL_CACHE_DIR / f"lecture_{lecture_number}.json"


def load_cached_summaries(lecture_number, expected_fingerprint):
    """Returns {chunk_index: {"summary", "usage"}} for the chunking we're
    currently working with, or {} if there's nothing reusable.

    Also migrates the pre-refactor cache format, which stored the full plain and
    contextual chunk objects and had no fingerprint. Those are adopted only if
    their plain chunk texts hash to the same value as the current chunking -
    which is exactly how lectures 26/42/43 get identified as needing a rebuild,
    with no hardcoded list of broken lectures anywhere.
    """
    path = cache_path(lecture_number)
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        cached = json.load(f)

    if "chunking_fingerprint" in cached:
        if cached["chunking_fingerprint"] != expected_fingerprint:
            return {}
        return {int(k): v for k, v in cached["summaries"].items()}

    # legacy format: {"plain": [...], "contextual": [...]}
    if cached.get("chunking_fingerprint") is None and "contextual" in cached:
        if fingerprint(cached.get("plain", [])) != expected_fingerprint:
            return {}
        return {
            c["chunk_index"]: {"summary": c["summary"], "usage": c.get("usage", {})}
            for c in cached["contextual"]
        }

    return {}


def save_cached_summaries(lecture_number, fp, summaries):
    config.CONTEXTUAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "lecture_number": lecture_number,
        "chunking_fingerprint": fp,
        "summaries": {str(k): v for k, v in sorted(summaries.items())},
    }
    with open(cache_path(lecture_number), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --- planning ---------------------------------------------------------------

def plan(plain_chunks, lectures):
    """Returns (cached, units) - what's already paid for, and one work unit per
    lecture that still needs summaries."""
    by_lecture = artifacts.group_by_lecture(plain_chunks)
    lecture_text = {l["lecture_number"]: l["text"] for l in lectures}

    cached, units = {}, []
    for n, chunks in sorted(by_lecture.items()):
        fp = fingerprint(chunks)
        have = load_cached_summaries(n, fp)
        cached[n] = {"fingerprint": fp, "summaries": have}

        missing = [c for c in chunks if c["chunk_index"] not in have]
        if not missing:
            continue

        items = []
        for c in missing:
            prompt = build_prompt(lecture_text[n], c["canonical_title"], c["text"])
            items.append({
                "lecture_number": n,
                "chunk_index": c["chunk_index"],
                "prompt": prompt,
                "prompt_tokens": config.count_tokens(prompt),
            })
        units.append({
            "lecture_number": n,
            "items": items,
            "total_chunks": len(chunks),
            "estimated_tokens": sum(it["prompt_tokens"] for it in items),
        })
    return cached, units


def pack_waves(units):
    """Greedy bin-packing of whole lectures into waves under the enqueued-token cap."""
    waves, current, current_tokens = [], [], 0
    for unit in sorted(units, key=lambda u: -u["estimated_tokens"]):
        if unit["estimated_tokens"] > config.WAVE_TOKEN_BUDGET:
            waves.append([unit])  # single lecture over budget - guard, shouldn't happen
            continue
        if current_tokens + unit["estimated_tokens"] > config.WAVE_TOKEN_BUDGET and current:
            waves.append(current)
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit["estimated_tokens"]
    if current:
        waves.append(current)
    return waves


# --- batch execution --------------------------------------------------------

def submit_wave(wave_units, wave_index):
    config.BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = config.BATCH_INPUT_DIR / f"wave_{wave_index}.jsonl"

    total_items = 0
    with open(input_path, "w", encoding="utf-8") as f:
        for unit in wave_units:
            for item in unit["items"]:
                body = {
                    "model": config.SUMMARY_MODEL,
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "temperature": 0,
                }
                f.write(json.dumps({
                    "custom_id": f"{item['lecture_number']}_{item['chunk_index']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }) + "\n")
                total_items += 1

    lecture_numbers = [u["lecture_number"] for u in wave_units]
    est = sum(u["estimated_tokens"] for u in wave_units)
    print(f"\nWave {wave_index}: lectures {lecture_numbers}, {total_items} chunks, ~{est:,} enqueued tokens")

    uploaded = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h"
    )
    print(f"  submitted: {batch.id}")
    return batch.id


def poll_until_done(batch_id):
    wait = config.POLL_START_SECONDS
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  [{time.strftime('%H:%M:%S')}] status: {batch.status}  counts: {batch.request_counts}")
        if batch.status == "completed":
            return batch
        if batch.status in ("failed", "expired", "cancelled"):
            print(f"  wave did not complete: {batch.status}. Errors: {batch.errors}")
            return batch
        time.sleep(wait)
        wait = min(wait * config.POLL_BACKOFF_FACTOR, config.POLL_MAX_SECONDS)


def finalize_wave(batch, wave_units, cached):
    results = {}
    for line in client.files.content(batch.output_file_id).text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        body = record["response"]["body"]
        usage = body["usage"]
        details = usage.get("prompt_tokens_details") or {}
        results[record["custom_id"]] = {
            "summary": body["choices"][0]["message"]["content"].strip(),
            "usage": {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "cached_tokens": details.get("cached_tokens", 0) or 0,
            },
        }

    for unit in wave_units:
        n = unit["lecture_number"]
        summaries = dict(cached[n]["summaries"])
        for item in unit["items"]:
            result = results.get(f"{n}_{item['chunk_index']}")
            if result is None:
                print(f"  WARNING: no result for lecture {n} chunk {item['chunk_index']}")
                continue
            summaries[item["chunk_index"]] = result
        cached[n]["summaries"] = summaries
        save_cached_summaries(n, cached[n]["fingerprint"], summaries)
        print(f"  lecture {n}: cached {len(summaries)}/{unit['total_chunks']} summaries")


# --- output -----------------------------------------------------------------

def assemble(plain_chunks, cached):
    contextual, incomplete = [], []
    for c in plain_chunks:
        entry = cached[c["lecture_number"]]["summaries"].get(c["chunk_index"])
        if entry is None:
            incomplete.append((c["lecture_number"], c["chunk_index"]))
            continue
        combined = entry["summary"] + "\n\n" + c["text"]
        contextual.append({
            "strategy": STRATEGY,
            "chunk_index": c["chunk_index"],
            "lecture_number": c["lecture_number"],
            "canonical_title": c["canonical_title"],
            "youtube_url": c["youtube_url"],
            "summary": entry["summary"],
            "text": combined,
            "word_count": config.count_words(combined),
            "token_count": config.count_tokens(combined),
            "usage": entry.get("usage", {}),
        })
    return contextual, incomplete


def total_cost(contextual):
    total = 0.0
    for c in contextual:
        u = c.get("usage") or {}
        if u:
            total += config.summary_cost(
                config.SUMMARY_MODEL, u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cached_tokens", 0)
            )
    return total


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report the plan and its cost, submit nothing")
    parser.add_argument("--force", action="store_true", help="discard ALL cached summaries and redo the whole corpus")
    args = parser.parse_args()

    plain_chunks = artifacts.read_chunks(config.PLAIN_CHUNKS_PATH)
    lectures = artifacts.load_lectures(config.TRANSCRIPTS_PATH)

    if args.force:
        removed = 0
        for path in config.CONTEXTUAL_CACHE_DIR.glob("lecture_*.json"):
            path.unlink()
            removed += 1
        print(f"--force: cleared {removed} cached lectures, everything will be re-summarized.")

    cached, units = plan(plain_chunks, lectures)

    reusable = sum(len(v["summaries"]) for v in cached.values())
    pending = sum(len(u["items"]) for u in units)
    print(f"{len(plain_chunks)} chunks total: {reusable} already summarized, {pending} pending "
          f"across {len(units)} lecture(s).")

    if units:
        waves = pack_waves(units)
        est_tokens = sum(u["estimated_tokens"] for u in units)
        est_cost = est_tokens / 1e6 * config.MODEL_RATES[config.SUMMARY_MODEL]["input"]
        print(f"Lectures needing work: {sorted(u['lecture_number'] for u in units)}")
        print(f"Packed into {len(waves)} wave(s), ~{est_tokens:,} enqueued tokens, "
              f"~${est_cost:.2f} at batch input rates (before prompt-cache discount).")

    if args.dry_run:
        print("\n--dry-run: nothing submitted.")
        return

    if units:
        for i, wave in enumerate(pack_waves(units)):
            batch = poll_until_done(submit_wave(wave, i))
            if batch.status != "completed":
                print(f"Stopping after wave {i}. Re-run this command to retry - "
                      f"finished lectures are cached and will not be repaid for.")
                return
            finalize_wave(batch, wave, cached)
        print("\nAll waves complete.")

    contextual, incomplete = assemble(plain_chunks, cached)
    if incomplete:
        print(f"WARNING: {len(incomplete)} chunks still have no summary, first: {incomplete[0]}. "
              f"Not writing the artifact - re-run to finish them.")
        return

    spent = total_cost(contextual)
    artifacts.write_chunks(
        config.CONTEXTUAL_CHUNKS_PATH,
        stage="contextualize",
        strategy=STRATEGY,
        chunks=contextual,
        summary_model=config.SUMMARY_MODEL,
        summary_cost_usd=round(spent, 4),
    )
    print(f"Wrote {len(contextual)} contextual chunks -> {config.CONTEXTUAL_CHUNKS_PATH}")
    print(f"Cumulative summarization cost recorded in the artifact: ${spent:.4f}")


if __name__ == "__main__":
    main()
