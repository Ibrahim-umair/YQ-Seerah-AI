# Code Walkthrough — the `seerah` package and the Makefile

This is a file-by-file, line-by-line explanation of every piece of code in the
pipeline as it exists today — everything under `seerah/`, plus the `Makefile`
that drives it. It does not cover `pilot_evaluation/`, a frozen historical
experiment kept as evidence, not part of the live pipeline.

Written to be read once, end to end, before starting on the agent layer — so
the retrieval foundation the agent will sit on is fully understood first.

## How the pieces fit together

```
data/seerah_transcripts.jsonl (104 lectures)
        |
        v
seerah/ingest/chunk.py            (Stage 1 - free, local)
        |  uses data/chunking_manifest.json to reproduce historical chunk boundaries
        v
data/chunks_plain.json  (2,763 sentence-aligned chunks)
        |
        v
seerah/ingest/contextualize.py    (Stage 2 - costs money, uses OpenAI Batch API)
        |  prepends an LLM-written summary to every chunk
        v
data/chunks_contextual.json
        |
        +---------------------------+
        v                           v
seerah/ingest/embed.py       seerah/ingest/bm25.py
(Stage 3 - costs money)      (Stage 4 - free, local)
        |                           |
        v                           v
Qdrant collection            data/bm25_index/
"seerah_full_corpus_contextual"
        |                           |
        +------------+--------------+
                     v
            seerah/retrieve.py
   (Retriever class: vector_search + bm25_search,
    both returning the same Hit dataclass)
                     |
        +------------+------------------+
        v                               v
  seerah/cli.py                  seerah/eval/run_retrieval.py
  (manual interactive              (checks retrieval quality
   querying, no eval)               against the 304-question set)
```

`seerah/config.py` and `seerah/artifacts.py` are not in this chain — every
other file imports from them. `config` holds every path/model/constant;
`artifacts` holds the JSON read/write helpers shared by every stage.

`seerah/eval/validate_questions.py` checks the question set itself
(`data/eval_questions_raw.json`) for integrity; it doesn't touch the
retrieval indexes at all.

Every ingestion stage follows the same rule, enforced by convention rather
than a shared base class: **if the stage's output already exists, do
nothing.** `--force` overrides that. This is what makes the whole pipeline
cheap to reproduce from a fresh clone — the two stages that cost real money
(contextualize, embed) are skipped by default whenever their artifact is
already sitting in the repository or already built locally.

---

## `seerah/__init__.py` (19 lines)

**Purpose.** The package's front door. Carries no logic — its docstring is
the single-paragraph map of the whole ingestion pipeline, so `help(seerah)`
or just opening this file tells you the four stage commands and their order
before you've read anything else.

**Line by line.**
- `L1-17` — module docstring. States what the package is (a RAG app over the
  104-lecture Seerah series), then lists the four `python -m seerah.ingest.*`
  commands in pipeline order with a one-line description each and a `$`
  marker on the two that cost money. Explains that every stage is a no-op if
  its output exists (`--force` to override), and ends with the query command
  (`python -m seerah.cli`).
- `L19` — `__version__ = "0.1.0"`. A conventional version string; nothing in
  the codebase currently reads it.

---

## `seerah/config.py` (79 lines)

**Purpose.** Every path, model name, and tunable constant that more than one
stage needs to agree on lives here — the fix for the earlier problem where
standalone scripts each hardcoded their own paths and quietly drifted apart
(the `seerah_transcripts.jsonl` path mismatch, the `BGE_VARIANTS`/BM25
directory coupling). Anything imported as `from seerah import config` gets
the same values everywhere.

**Line by line.**
- `L1-6` — docstring: one line stating the file's purpose, one line
  explaining *why* it exists (prevents stage drift).
- `L8-9` — `import sys` (needed for stdout reconfiguration later) and
  `from pathlib import Path` (cross-platform path construction — this is
  part of why the pipeline works unmodified on Windows).
- `L11-12` — `import tiktoken` (OpenAI's exact tokenizer, so token counts here
  match what OpenAI's API actually bills/limits on) and
  `from dotenv import load_dotenv`.
- `L14` — `load_dotenv()` runs at **import time**. The instant anything does
  `from seerah import config`, the `.env` file's contents (like
  `OPENAI_API_KEY`) are loaded into `os.environ`, where the `openai` SDK picks
  them up automatically without any code here having to pass them explicitly.
- `L16` — `REPO_ROOT = Path(__file__).resolve().parent.parent`. Walks up two
  directories from this file (`config.py` → `seerah/` → repo root). Every
  path below is anchored to this, so scripts work correctly regardless of
  what directory they're actually invoked from.
- `L17` — `DATA_DIR = REPO_ROOT / "data"`.
- `L20-23` — the four core artifact paths: raw transcripts
  (`TRANSCRIPTS_PATH`), the chunking manifest that records which lecture used
  which metadata mode (`MANIFEST_PATH`), stage 1's output
  (`PLAIN_CHUNKS_PATH`), stage 2's output (`CONTEXTUAL_CHUNKS_PATH`).
- `L25-27` — three directories: `CONTEXTUAL_CACHE_DIR` (stage 2's per-lecture
  resume cache, gitignored), `BATCH_INPUT_DIR` (the `.jsonl` files actually
  submitted to OpenAI's Batch API, gitignored), `BM25_DIR` (stage 4's
  persisted index, gitignored).
- `L30-31` — `CHUNK_SIZE = 800`, `CHUNK_OVERLAP = 80`: the SentenceSplitter
  settings chosen from the pilot evaluation (see the README's chunking
  section for the comparison against fixed-window splitting).
- `L34-36` — `SUMMARY_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`. The
  embedding dimension (3072) must match the vector size the Qdrant collection
  is created with in `embed.py` — if this ever changes, the collection has to
  be rebuilt from scratch, not just re-populated.
- `L39-41` — `QDRANT_URL`, `COLLECTION_NAME` (the one collection the whole
  pipeline uses), `EMBED_BATCH_SIZE = 50` (chunks per OpenAI embeddings call
  in stage 3 — batching amortizes network round-trip cost).
- `L43-50` — Batch API tuning, with a comment explaining *why*: a single
  contextual-summary request carries the entire ~17k-token lecture as
  context, so the full corpus sums to ~26M enqueued tokens against OpenAI's
  2M cap. `WAVE_TOKEN_BUDGET = 1_800_000` stays under that cap with
  headroom. `POLL_START_SECONDS`/`POLL_MAX_SECONDS`/`POLL_BACKOFF_FACTOR`
  drive the exponential-backoff loop that waits for a submitted batch job.
- `L52-53` — `MODEL_RATES`: nested dict of per-million-token USD pricing for
  `gpt-5.4-nano`'s three token categories (fresh input, prompt-cache-discounted
  input, output) at Batch API rates (comment notes 50% off standard pricing).
  `EMBEDDING_RATE`: flat per-million-token price for `text-embedding-3-large`.
- `L56` — `TOKENIZER = tiktoken.get_encoding("cl100k_base")`, loaded once at
  import time rather than per call (loading a tokenizer has real setup cost).
- `L59-60` — `count_tokens(text)`: one-liner, `len(TOKENIZER.encode(text))`.
  Used everywhere a token budget needs to be measured — chunking, wave
  packing, cost estimates.
- `L63-64` — `count_words(text)`: naive `len(text.split())`. Used for the
  human-readable `word_count` field stored alongside the more precise
  `token_count` on every chunk.
- `L67-74` — `summary_cost(model, input_tokens, output_tokens,
  cached_tokens=0)`: splits `input_tokens` into "fresh" (full price) and
  "cached" (OpenAI's automatic prompt-cache discount) portions, prices each
  separately, adds output token cost. This is what turns raw per-call
  `usage` numbers into the actual `$1.4762` figure recorded in the
  contextual-chunks artifact — a measured cost, not an estimate.
- `L77-79` — `use_utf8_stdout()`: reconfigures stdout to UTF-8 with
  `errors="replace"`. Docstring explains why — Windows consoles default to
  cp1252 and crash the instant Arabic script or the `ﷺ` glyph hits the
  console. Every CLI entry point (`chunk.py`, `contextualize.py`, `embed.py`,
  `bm25.py`, `cli.py`, `run_retrieval.py`) calls this as the first line of
  `main()`.

---

## `seerah/artifacts.py` (60 lines)

**Purpose.** The read/write contract for the JSON files passed between
stages. Centralizes serialization so no stage reinvents it, and tolerates
older artifact shapes left over from before this refactor.

**Line by line.**
- `L1-6` — docstring.
- `L8` — `import json` — the file's only dependency.
- `L11-23` — `write_chunks(path, stage, strategy, chunks, **extra)`:
  - `L12-19` — builds a `payload` dict: self-describing metadata first
    (`stage` name, `strategy` label, `num_chunks`, `num_lectures` — the
    latter computed via a set comprehension over `chunks` so it's always
    accurate even if chunks arrive out of lecture order), then splices in
    whatever `**extra` keyword args the caller passed (stage 1 passes
    `chunk_size`/`chunk_overlap`; stage 2 passes `summary_model`/
    `summary_cost_usd`), and finally the actual `chunks` list last.
  - `L20` — `path.parent.mkdir(parents=True, exist_ok=True)`: creates the
    destination directory if it doesn't exist yet, so a first run on a fresh
    clone doesn't fail on a missing `data/` subfolder.
  - `L21-22` — writes with `ensure_ascii=False` (so Arabic script and `ﷺ` are
    stored as literal UTF-8 characters rather than escaped `\uXXXX`
    sequences — much more readable in the committed file) and `indent=2`
    (human-diffable in git).
  - `L23` — returns the payload dict too, so a caller doesn't have to
    re-read the file if it wants the data right after writing it.
- `L26-41` — `read_chunks(path)`:
  - `L29-33` — raises `FileNotFoundError` with an actionable message
    (run the producing stage, or pull the committed artifact) rather than a
    bare "file not found."
  - `L34-35` — loads the JSON.
  - `L36-37` — the current-format path: if `"chunks"` is a top-level key
    (what `write_chunks` produces), return that list.
  - `L38-40` — legacy fallback: checks for the two pre-refactor top-level key
    names (`"Recursive + Contextual"`, `"Recursive (Sentence)"`), so old
    artifacts on disk never needed manual conversion.
  - `L41` — if neither shape matches, raises `ValueError` listing the actual
    top-level keys found, for debugging.
- `L44-51` — `load_lectures(transcripts_path)`: reads the `.jsonl` file line
  by line, skipping blank lines, `json.loads`-ing each into a dict, then
  **sorts by `lecture_number`** before returning — guaranteeing every caller
  sees lectures in stable numeric order regardless of on-disk line order.
- `L54-60` — `group_by_lecture(chunks)`: builds `{lecture_number: [chunks]}`
  via `setdefault`, then sorts each lecture's chunk list by `chunk_index` in
  place. Used by the chunking coverage check and by stage 2's fingerprinting,
  anywhere code needs "all of lecture N's chunks, in order."

---

## `seerah/ingest/__init__.py` (11 lines)

**Purpose.** Pure documentation, no code. Lists the four stage commands with
their I/O and cost split (stages 1 and 4 free/local, 2 and 3 cost money),
and notes that every stage skips its work if the output already exists —
pointing at the README for the full reproducibility story.

---

## `seerah/ingest/chunk.py` (173 lines) — Stage 1

**Purpose.** Splits the 104 raw transcripts into ~2,763 sentence-aligned
chunks via LlamaIndex's `SentenceSplitter`. This is the stage where a real
historical data-loss bug happened (lectures 26/42/43 lost 2,201 characters
to a resume-logic mismatch), so it carries the most defensive checking of
any stage — a coverage check and a byte-for-byte verify mode.

**Line by line.**
- `L1-25` — docstring: what it does and why (sentence-aware vs. fixed-window
  splitting — a chunk never begins or ends mid-sentence), I/O paths, and a
  detailed explanation of the `chunking_manifest.json` mechanism: LlamaIndex's
  `SentenceSplitter` reserves token budget for a `Document`'s metadata, so
  passing populated metadata shifts every chunk boundary; earlier runs of
  this project did that and later runs didn't, so the manifest records which
  mode produced each lecture, letting this stage reproduce the committed
  artifact byte-for-byte instead of silently re-cutting the corpus. States
  the three usage modes.
- `L27-33` — imports: `argparse`, `json` (for the manifest), LlamaIndex's
  `Document` and `SentenceSplitter`, local `artifacts`/`config`.
- `L35` — `STRATEGY = "Recursive (Sentence)"` — a label stamped onto every
  chunk's `"strategy"` field and the artifact's top-level metadata.
- `L38-41` — `load_manifest()`: opens `chunking_manifest.json`, returns just
  its `"lectures"` sub-dict (lecture-number-as-string → mode-string).
- `L44-68` — `chunk_lecture(lecture, splitter, mode)`:
  - `L46-52` — if `mode == "with_metadata"`, builds a metadata dict with
    `lecture_number`/`canonical_title`/`youtube_url`; otherwise metadata
    stays `{}`. **This one branch is the entire historical cause of the
    boundary inconsistency** — SentenceSplitter's token accounting changes
    depending on whether this dict is populated.
  - `L53` — wraps the lecture's full text plus that metadata into a
    LlamaIndex `Document`.
  - `L54` — `splitter.get_nodes_from_documents([doc])` runs the actual
    splitting algorithm, returning a list of `TextNode`s.
  - `L56-68` — converts each node into this project's chunk schema:
    strategy label, its index within the lecture (`chunk_index`), the
    lecture's identifying fields copied across, the chunk's text
    (`node.get_content()`), and both `word_count`/`token_count` via
    `config`'s helpers.
- `L71-80` — `build_chunks()`: loads all lectures, loads the manifest,
  constructs one `SentenceSplitter` (reused across all 104 lectures, not
  rebuilt per lecture), then loops, looking up each lecture's mode from the
  manifest — **defaulting to `"without_metadata"`** for any lecture number
  the manifest doesn't mention, meaning any lecture added in the future
  automatically gets the full 800-token budget. Returns both `lectures` and
  `chunks`, since the coverage check needs the raw lecture text too.
- `L83-120` — `report_coverage(lectures, chunks)` — **the safety net that
  would have caught the historical bug**:
  - `L88` — regroups the freshly-built chunks by lecture.
  - `L91-100` — for each lecture, locates where each of its chunks' text
    actually sits inside the original transcript string via `text.find`,
    building `(start, end)` character spans; warns (without crashing) if a
    chunk's text can't be found verbatim.
  - `L101` — sorts spans by start position so gaps can be measured in order.
  - `L103` — initializes `missing` to the first span's start position — if
    the transcript's opening characters aren't covered by any chunk, that's
    already counted.
  - `L104-107` — walks consecutive span pairs; a gap larger than 1 character
    between where one chunk ends and the next begins gets added to
    `missing` (a 1-character gap is just the space between sentences, not a
    real hole). **This is exactly the check that would have caught the
    1,379/337/485-character gaps** in lectures 26/42/43.
  - `L108-110` — also checks the tail: if the last chunk doesn't reach the
    transcript's end, that trailing text counts as missing too.
  - `L112-114` — prints a per-lecture warning if it has any missing
    characters, and accumulates a running `total_missing`.
  - `L116-119` — prints a clean success line if `total_missing == 0`, or a
    failure line telling the user to investigate.
  - `L120` — returns the total (currently only used for printing — the
    stage doesn't hard-fail on a coverage gap, worth knowing if this is ever
    wired into CI).
- `L123-137` — `verify_against_artifact(chunks)`: loads the currently
  committed `chunks_plain.json`, compares lengths first (cheap), then
  compares every chunk's `text` field in order; reports the first mismatch's
  `(lecture_number, chunk_index)` if any differ, otherwise prints a VERIFIED
  message. This is what proves the chunking is deterministic.
- `L140-173` — `main()`:
  - `L141` — `use_utf8_stdout()`.
  - `L142-145` — argparse: `--force` (rebuild even if output exists),
    `--verify` (diff without writing).
  - `L147-149` — `--verify` path: re-chunk in memory only, exit 0/1 based on
    the match — never touches disk.
  - `L151-155` — the default "use committed data" path (mode B): if the
    output already exists and `--force` wasn't passed, report the existing
    count and how to force/verify, then return.
  - `L157-159` — otherwise actually chunk and run the coverage check.
  - `L161-168` — writes via `artifacts.write_chunks`, passing
    `chunk_size`/`chunk_overlap` as extra metadata.
  - `L169` — final summary line.
  - `L172-173` — `if __name__ == "__main__": main()`.

---

## `seerah/ingest/contextualize.py` (360 lines) — Stage 2, the most complex file

**Purpose.** For every chunk, get an LLM to read the *entire* lecture plus
that one chunk and write a 1-2 sentence summary, then prepend it to the
chunk before embedding/indexing — Anthropic's Contextual Retrieval
technique. Uses OpenAI's Batch API (cheaper, separate rate-limit pool) with
a resume-safe caching scheme keyed on a content fingerprint rather than a
chunk index — the actual fix for the lecture 26/42/43 data-loss bug.

**Line by line, grouped by the file's own section comments.**

- `L1-35` — docstring: explains the technique and motivation, I/O paths, a
  cost warning ($1.47/several hours from scratch, with `--dry-run` to
  preview first), why the Batch API specifically (each request carries the
  ~17k-token lecture as context, so the sync API's 200k TPM limit is hit
  constantly; Batch has a separate 2M-token *enqueued* cap and is 50%
  cheaper), and the fingerprint-based resume-safety explanation naming the
  exact bug it fixes (2,201 characters lost across 3 lectures when the old
  design resumed by chunk index instead of content).
- `L37-44` — imports: `argparse`, `hashlib` (SHA-256 fingerprinting), `json`,
  `time` (polling sleep), OpenAI's client class, local `artifacts`/`config`.
- `L46` — `STRATEGY = "Recursive + Contextual"`.
- `L48` — `client = OpenAI()`, constructed once at import time (reads
  `OPENAI_API_KEY` from the environment `config` already populated).
- `L51-59` — `fingerprint(lecture_chunks)`: SHA-256 over every chunk's text
  (UTF-8 encoded) with a null-byte separator between chunks, in order,
  returning the hex digest. Because it hashes literal **text**, not
  position, any change to how a lecture is chunked — a different splitter
  setting, a repaired transcript, a metadata-mode flip — produces a
  different fingerprint. This is the single property that makes stale-cache
  detection reliable.
- `L62-70` — `build_prompt(lecture_text, canonical_title, chunk_text)`:
  constructs the literal prompt — states the lecture title, wraps the full
  transcript in `<lecture>` tags, wraps the specific chunk in `<chunk>`
  tags, instructs the model to answer with only a 1-2 sentence
  self-contained summary. This whole prompt (full lecture + one chunk) is
  why each request is so token-heavy, which is the root cause of the
  wave-packing logic further down.
- `L75-76` — `cache_path(lecture_number)`: returns e.g.
  `data/contextual_cache/lecture_46.json`.
- `L79-110` — `load_cached_summaries(lecture_number, expected_fingerprint)`:
  - `L89-91` — no cache file yet → return `{}`.
  - `L93-94` — load the cached JSON.
  - `L96-99` — **current format**: if `"chunking_fingerprint"` is present,
    compare it against `expected_fingerprint`; mismatch → `{}` (forces full
    re-summarization of this lecture); match → return the summaries dict
    with integer keys (JSON only allows string keys, so this converts back).
  - `L101-108` — **legacy format**: handles the pre-refactor cache shape
    (`{"plain": [...], "contextual": [...]}`, no fingerprint field at all).
    Computes a fingerprint over the legacy cache's own `"plain"` chunk list
    and compares it against `expected_fingerprint`. Match → extract
    `{chunk_index: {summary, usage}}` from the legacy `"contextual"` list.
    **This is precisely the mechanism** that let all 101 unaffected
    lectures' pre-existing summaries be adopted for free while
    automatically identifying lectures 26/42/43 as needing a redo (their
    legacy fingerprints no longer matched the newly-fixed chunking) — with
    no hardcoded list of "broken" lecture numbers anywhere in the code.
  - `L110` — falls through to `{}` if neither shape matched.
- `L113-121` — `save_cached_summaries(lecture_number, fp, summaries)`:
  ensures the cache directory exists, writes `{lecture_number,
  chunking_fingerprint, summaries}` (keys stringified) to that lecture's
  cache file.
- `L126-157` — `plan(plain_chunks, lectures)` — the heart of the
  incremental/resumable design:
  - `L129` — groups all plain chunks by lecture.
  - `L130` — builds `lecture_number → full text` lookup (needed for
    prompts).
  - `L133-136` — for every lecture (sorted, deterministic order): computes
    its current fingerprint, tries loading cached summaries under that
    fingerprint, records both in a `cached` dict.
  - `L138-140` — finds which of this lecture's chunks still lack a summary;
    if none are missing, `continue` — no work needed for this lecture.
  - `L142-150` — for chunks that ARE missing, builds one work item per
    chunk: lecture number, chunk index, the prebuilt prompt, its token
    count (needed for wave-packing).
  - `L151-156` — if there was any missing work, appends a "unit" (one
    lecture's pending work) recording the lecture number, its items, its
    total chunk count, and the summed token count of its items
    (`estimated_tokens`) for bin-packing.
  - `L157` — returns `(cached, units)`.
- `L160-174` — `pack_waves(units)` — greedy bin-packing:
  - `L163` — sorts units largest-first (a standard bin-packing heuristic —
    tends to produce fewer bins than smallest-first).
  - `L164-166` — a single lecture exceeding the whole wave budget alone gets
    forced into its own wave — a defensive guard the docstring notes
    "shouldn't happen" given known lecture lengths, but handled rather than
    assumed away.
  - `L167-169` — if adding the next unit would exceed the budget AND the
    current wave already has something in it, close the current wave and
    start a new one.
  - `L170-171` — add the unit to whichever wave is open, track running
    tokens.
  - `L172-173` — append the final non-empty wave after the loop ends.
  - `L174` — returns the list of waves.
- `L179-209` — `submit_wave(wave_units, wave_index)`:
  - `L180-181` — ensures the batch-input directory exists, computes this
    wave's `.jsonl` path.
  - `L184-198` — writes one JSON line per pending chunk in OpenAI's Batch
    API input format: `custom_id` = `"{lecture}_{chunk_index}"` (used later
    to match results back), fixed `method`/`url` for the chat completions
    endpoint, `body` with the model name, a single user message containing
    the prebuilt prompt, `temperature=0`.
  - `L200-202` — prints which lectures are in this wave, chunk-request
    count, estimated tokens.
  - `L204` — uploads the `.jsonl` file with `purpose="batch"`.
  - `L205-207` — creates the batch job against that file, the chat
    completions endpoint, and a 24-hour completion window (OpenAI's max —
    gives the job the best chance at the cheaper batch tier).
  - `L208` — returns the batch ID.
- `L212-223` — `poll_until_done(batch_id)`: exponential backoff — starts at
  `POLL_START_SECONDS` (30s), each iteration retrieves status and request
  counts, prints a timestamped line, returns immediately on `"completed"`,
  also returns (without raising) on a terminal failure state (`failed`/
  `expired`/`cancelled`) so the caller can decide what to do; otherwise
  sleeps `wait` seconds then multiplies `wait` by the backoff factor
  (capped at `POLL_MAX_SECONDS`, 10 minutes).
- `L226-255` — `finalize_wave(batch, wave_units, cached)` — turns a
  completed batch's raw output into updated caches:
  - `L228` — downloads the whole output file as text (JSONL, one line per
    request).
  - `L229-242` — for each non-blank line: parses the record, pulls the
    response body, extracts the stripped summary text, pulls token usage
    including any `cached_tokens` (OpenAI's automatic prompt-cache discount,
    defaulting to 0 if absent), stores it all in a `results` dict keyed by
    `custom_id`.
  - `L244-255` — for each unit (lecture) in this wave: starts from a *copy*
    of that lecture's already-cached summaries, then for every pending item
    looks up its result by reconstructing its `custom_id`; missing results
    are warned about and skipped (defensive — shouldn't normally happen);
    otherwise adds the new summary to the running dict. After all items,
    updates `cached[n]["summaries"]` and **immediately persists to disk**
    via `save_cached_summaries` — so a crash right after this point loses at
    most the in-flight batch, never previously-finalized work. Prints a
    per-lecture progress line.
- `L260-280` — `assemble(plain_chunks, cached)`: for every plain chunk in
  original order, looks up its summary; if missing, records it in an
  `incomplete` list rather than silently dropping it; otherwise builds
  `combined = summary + "\n\n" + original_text` and the final contextual
  chunk dict (same identifying fields, plus `summary`, `text` = combined,
  word/token counts computed over the combined text, and the raw `usage`
  stats preserved for cost accounting). Returns `(contextual, incomplete)`.
- `L283-291` — `total_cost(contextual)`: sums `config.summary_cost(...)`
  over every chunk's recorded `usage`, skipping any chunk with none. Turns
  per-chunk usage numbers into the single `$1.4762` figure.
- `L294-360` — `main()`:
  - `L295` — UTF-8 stdout.
  - `L296-299` — argparse: `--dry-run` (report plan/cost, submit nothing),
    `--force` (wipe all cached summaries, redo everything).
  - `L301-302` — loads the plain chunks artifact and raw transcripts.
  - `L304-309` — `--force`: deletes every `lecture_*.json` cache file,
    reports the count removed.
  - `L311` — computes the plan (`cached`, `units`).
  - `L313-316` — prints total chunks, how many are reusable, how many
    pending, across how many lectures.
  - `L318-324` — if there's pending work: bin-packs into waves, estimates
    tokens and cost at Batch input rates (noted as *before* the prompt-cache
    discount that will actually apply), prints which lectures need work.
    This block is exactly what `--dry-run` shows.
  - `L326-328` — `--dry-run` stops here; nothing submitted.
  - `L330-338` — otherwise, for each wave (recomputed via `pack_waves` — a
    second call, but harmless since `--dry-run` already returned if that
    flag was set): submit, poll to completion, and if the batch didn't
    complete successfully, print a "safe to re-run" message and return
    early rather than pushing forward into an inconsistent state.
  - `L340-344` — after all waves succeed, assembles the final chunk list;
    if anything is still incomplete (shouldn't happen if waves all
    completed, but checked anyway), warns and refuses to write a partial
    artifact.
  - `L346-355` — computes total spend, writes the final artifact via
    `artifacts.write_chunks` with `summary_model`/`summary_cost_usd` as
    extra metadata, prints the chunk count, path, and cost.
  - `L359-360` — entry-point guard.

---

## `seerah/ingest/embed.py` (158 lines) — Stage 3

**Purpose.** Embeds every contextual chunk with `text-embedding-3-large` and
loads vectors + payload into a Qdrant collection. Cheap (~$0.28 for the
whole corpus). This is the one stage a fresh clone realistically must run,
because the vector store itself is too large/impractical to commit — every
stage upstream of it is already a committed artifact.

**Line by line.**
- `L1-21` — docstring: I/O paths, cost, why this specific stage is the one a
  reviewer must run, the Docker requirement, why this embedding model was
  chosen (beat local BGE-M3 on every retrieval metric — see the README), the
  three usage modes.
- `L23-30` — imports: `argparse`, OpenAI client, Qdrant client plus its
  `Distance`/`PointStruct`/`VectorParams` model classes, local
  `artifacts`/`config`.
- `L31` — module-level `client = OpenAI()`.
- `L34-43` — `connect()`: tries constructing a `QdrantClient` and calling
  `get_collections()` as a connectivity probe; any exception → exits with a
  clear message pointing at `docker compose up -d`, rather than surfacing a
  raw connection-refused traceback.
- `L46-92` — `verify(qdrant, chunks)` — hardened during this project after
  discovering a naive check wasn't enough:
  - `L56-58` — collection doesn't exist → report and fail.
  - `L60-63` — compares the collection's `points_count` against the number
    of chunks in the artifact — a cheap first check.
  - `L65-81` — scrolls every point in pages of 1000 (Qdrant's `scroll` API
    returns `(points, next_offset)`, and `offset is None` signals the last
    page). For each point: missing/empty `text`/`lecture_number`/
    `chunk_index` → counted as `missing`; otherwise records the lecture
    number as "seen," and — the important part — checks whether the
    payload's stored `text` still matches `chunks[p.id]["text"]` in the
    **current** artifact, relying on the invariant that points were upserted
    with `id == index into the artifact list` (established in `build()`
    below). Mismatch, or `p.id` out of range/not an int → counted as
    `stale`. **This is the exact check that caught the 92 chunks whose text
    changed** after the lecture 26/42/43 repair while the total point count
    stayed at 2,763 — a bug a naive count-and-payload-presence check would
    have completely missed.
  - `L83-89` — prints specific messages for nonzero `missing`/`stale`
    counts; returns `False` if either is nonzero.
  - `L91-92` — only if nothing failed: prints VERIFIED (now explicitly
    saying "text matches the artifact") and returns `True`.
- `L95-132` — `build(qdrant, chunks)`:
  - `L96-98` — if the collection already exists, deletes it first — a full
    rebuild, not an incremental upsert, so stale and fresh points can never
    end up mixed in the same collection.
  - `L99-102` — creates a fresh collection with `EMBEDDING_DIM` (3072) and
    cosine distance (OpenAI's recommended metric for these embeddings).
  - `L104` — initializes a running token counter.
  - `L105-128` — loops over `chunks` in batches of `EMBED_BATCH_SIZE` (50):
    prints progress, calls the embeddings API once for the whole batch's
    texts (one call embeds up to 50 chunks — much cheaper than one call
    each), accumulates `response.usage.total_tokens`, builds one
    `PointStruct` per chunk in the batch with `id = start + i` (its absolute
    index into the full `chunks` list — **the invariant `verify()` relies
    on**), `vector` = that chunk's embedding, `payload` = identifying fields
    plus the full contextual `text` (so retrieval returns actual content
    without a second lookup). Upserts the batch.
  - `L130-131` — computes and prints total embedding cost, returns
    `(total_tokens, cost)` — though `main()` currently discards these return
    values, so the embedding cost isn't persisted into any artifact the way
    stage 2's cost is (a small loose end worth knowing about).
- `L135-158` — `main()`:
  - `L136` — UTF-8 stdout.
  - `L137-140` — argparse: `--force` (rebuild regardless), `--verify` (check
    only).
  - `L142-143` — loads the contextual chunks artifact, connects to Qdrant.
  - `L145-146` — `--verify`: run the check, exit 0/1, nothing else touched.
  - `L148-150` — default "use committed/already-built" path: if not
    `--force` and the existing collection already verifies healthy, report
    "nothing to do" and stop.
  - `L152-154` — otherwise, build and then immediately re-verify the
    freshly built collection as a sanity check.
  - `L157-158` — entry-point guard.

---

## `seerah/ingest/bm25.py` (66 lines) — Stage 4

**Purpose.** Builds a BM25 (keyword) index over the exact same contextual
chunk texts the vector store embeds, so the two retrieval methods are
compared/combined on equal footing rather than on different inputs. Free,
local, fast.

**Line by line.**
- `L1-15` — docstring: I/O, why it indexes the *contextual* text
  specifically, usage modes.
- `L17-23` — imports: `argparse`, `shutil` (removes an existing index
  directory on `--force`), LlamaIndex's `TextNode` and `BM25Retriever`,
  local `artifacts`/`config`.
- `L26-43` — `build(chunks)`:
  - `L27-39` — builds one `TextNode` per chunk: `text` = the chunk's
    contextual text, `id_` = its stringified index in the chunk list
    (LlamaIndex needs a string node ID; this mirrors the embedding stage's
    `id` convention though used slightly differently here), `metadata` =
    the same identifying fields (`lecture_number`, `canonical_title`,
    `youtube_url`, `chunk_index`) surfaced back at retrieval time.
  - `L40` — constructs a `BM25Retriever` via `from_defaults`, with
    `similarity_top_k=10` as its default (though `seerah/retrieve.py`
    overrides this per query anyway).
  - `L41-42` — ensures the output directory exists, then `.persist()`s the
    index to disk (LlamaIndex's own serialization format — the
    `corpus.jsonl`, `.npy` index files, `vocab.index.json` seen in
    `data/bm25_index/`).
  - `L43` — returns the node count.
- `L46-66` — `main()`:
  - `L47` — UTF-8 stdout.
  - `L48-50` — argparse: just `--force`.
  - `L52` — loads the contextual chunks artifact.
  - `L54-58` — if the output directory already exists and has content:
    without `--force`, report "already exists" and stop; with `--force`,
    `shutil.rmtree` it first (persist doesn't necessarily clean up stale
    index files from a differently-sized corpus on its own).
  - `L60-62` — builds the index, prints the final chunk count and output
    path.
  - `L65-66` — entry-point guard.

---

## `seerah/retrieve.py` (110 lines)

**Purpose.** The one shared retrieval layer both the CLI and the eventual
web app/agent will call — retrieval logic lives in exactly one place, behind
one result type, regardless of caller.

**Line by line.**
- `L1-6` — docstring: the "one place, two callers, same result shape"
  design intent.
- `L8-13` — imports: `time` (latency measurement), `dataclass` decorator,
  OpenAI client, Qdrant client, LlamaIndex's `BM25Retriever`, local
  `config`.
- `L17` — `DEFAULT_TOP_K = 10`.
- `L20-31` — `Hit` dataclass:
  - fields: `score` (retriever-specific — cosine similarity for vector, BM25
    score for keyword; **these two scores are not directly comparable**,
    worth knowing before any future reranking/fusion work), `lecture_number`,
    `canonical_title`, `youtube_url`, `chunk_index`, `text` (the full
    contextual chunk text).
  - `L29-31` — `citation` property: formats `"Lecture N: Title (url)"` on
    demand — what a future generation step would use for an inline
    citation.
- `L34-58` — `Retriever.__init__(self, load_bm25=True, load_vector=True)`:
  - `L39` — constructs `OpenAI()` only if vector search is wanted (avoids an
    unnecessary client when only BM25 is needed).
  - `L43-50` — if vector wanted: connects to Qdrant, explicitly checks
    `collection_exists`; if not, raises `SystemExit` naming the exact two
    fix commands (`docker compose up -d`, then the embed stage) — the same
    "always tell the user the fix" style used throughout the pipeline.
  - `L52-58` — if BM25 wanted: checks the index directory exists and is
    non-empty, raising a similarly actionable error if not; otherwise loads
    the persisted `BM25Retriever`.
  - The two `load_*` flags let a caller build a retrieval-only-one-way
    instance (used by `seerah.cli --retriever vector`), skipping the cost of
    loading an index that won't be used.
- `L60-62` — `embed_query(query)`: one-line wrapper around a single-string
  embeddings API call — factored out so both `vector_search` and any future
  caller (e.g. an agent doing its own query rewriting before embedding) can
  reuse it.
- `L64-90` — `vector_search(query, top_k=DEFAULT_TOP_K)`:
  - `L66-68` — times the embedding call specifically (`embed_seconds`) —
    this is the "OpenAI API round trip" cost the README flags as the real
    tradeoff of a remote embedding model over a local one.
  - `L70-77` — times the Qdrant query separately (`search_seconds`) —
    `query_points` with the query vector, requesting `top_k` results with
    payloads attached.
  - `L79-89` — converts Qdrant's raw `ScoredPoint`s into `Hit`s, pulling
    fields from `.payload` (using `.get("youtube_url", "")` defensively in
    case an older point predates that field).
  - `L90` — returns `(hits, embed_seconds, search_seconds)` — the 3-tuple
    behind the CLI's per-query timing breakdown.
- `L92-110` — `bm25_search(query, top_k=DEFAULT_TOP_K)`:
  - `L94` — mutates `self.bm25.similarity_top_k` before each call —
    LlamaIndex's `BM25Retriever` reads this attribute at retrieve time, so
    this is how one loaded retriever instance serves different `top_k`
    values across different queries without being rebuilt.
  - `L95-97` — times the retrieval call.
  - `L99-109` — converts LlamaIndex's `NodeWithScore` results into the same
    `Hit` shape, pulling fields from `.metadata` (BM25 stores identifying
    info as node metadata rather than a Qdrant-style payload) and getting
    text via `.get_content()`.
  - `L110` — returns `(hits, search_seconds)` — a 2-tuple (no separate embed
    step for pure keyword matching).

---

## `seerah/cli.py` (103 lines)

**Purpose.** The interactive, human-facing retrieval tool. Loads both
indexes once, then loops accepting free-typed queries, showing both
retrievers side by side with full chunk text and timing. Explicitly a
retrieval-inspection tool — **no generation happens here**.

**Line by line.**
- `L1-11` — docstring: what it does, usage examples with flags.
- `L13` — `import argparse`.
- `L15-17` — rich imports (`Console`, `Panel`, `Table`) for the terminal UI.
- `L19-20` — local `config` and `Retriever` imports.
- `L22` — module-level `console = Console()`.
- `L25-35` — `print_summary_table(title, hits)`: builds a rich `Table` with
  columns rank/score/lecture/title/chunk, one row per hit — the compact
  "scan the results" view.
- `L38-42` — `print_full_chunks(title, hits)`: for each hit, prints a rich
  `Panel` containing the full chunk text, titled with rank/score/
  citation/chunk index — the detailed "read the actual text" view.
- `L45-103` — `main()`:
  - `L46` — UTF-8 stdout.
  - `L47-52` — argparse: `--top-k` (default 10), `--retriever` (`both`/
    `vector`/`bm25` choice), `--full-text`/`--no-full-text` (a boolean pair
    that defaults to `True` but can be turned off).
  - `L55-56` — derives `want_vector`/`want_bm25` booleans from the
    `--retriever` choice.
  - `L58-59` — prints a loading message, constructs the shared `Retriever`
    loading only the wanted index/indexes.
  - `L60-63` — prints the "ready" banner with usage instructions.
  - `L65-73` — the input loop: reads a line via `console.input`, catching
    `EOFError`/`KeyboardInterrupt` for a graceful exit (Ctrl+D/Ctrl+C
    shouldn't produce a traceback); an empty line or "exit"/"quit"
    (case-insensitive) also ends the loop.
  - `L75-84` — for each query: conditionally runs vector search (recording
    embed+search timing) and/or BM25 search (recording its timing) based on
    the `want_*` flags.
  - `L86` — prints the combined timing line, joining whichever timing
    strings were collected.
  - `L88-91` — prints the compact summary table(s) for whichever
    retriever(s) ran.
  - `L93-97` — if `--full-text` (the default), also prints full-chunk-text
    panels.
  - `L99` — prints a separator before looping for the next query.
  - `L102-103` — entry-point guard.

---

## `seerah/eval/__init__.py` (10 lines)

**Purpose.** Pure docstring. States the question set lives at
`data/eval_questions_raw.json`, that "raw" means **pre-grounding** — every
question carries *candidate* supporting quotes proposed when it was
written, and an independent grounding pass still needs to confirm those
quotes and resolve them to exact chunk sets before they can be treated as
retrieval labels — and points at the `validate_questions` command.

---

## `seerah/eval/validate_questions.py` (176 lines)

**Purpose.** Integrity checker for the 304-question evaluation set: schema
completeness, tier consistency, verbatim quote checking against the real
transcripts, near-duplicate detection, meta-question detection. Backs `make
validate-questions` and `make verify`.

**Line by line.**
- `L1-23` — docstring: lists every check performed; explains why the
  verbatim-quote check matters most (a paraphrased/tidied quote can never
  be relocated and silently becomes an unlabeled question); explains that
  `expected_lectures` was deliberately **not** kept as a stored field once
  it was confirmed to always be exactly derivable from `supporting_quotes`
  — so this validator derives a question's lecture set live rather than
  trusting a second copy that could silently drift out of sync.
- `L25-30` — imports: `json`, `re` (meta-phrase regexes), `sys` (CLI arg for
  target path), `Counter`, `SequenceMatcher` from difflib (fuzzy string
  similarity for near-duplicate detection), `Path`, local `config`.
- `L34-35` — `REQUIRED`: the field set every question must have —
  `question_id`, `tier`, `cross_episode`, `question`, `reference_answer`,
  `supporting_quotes`. (Trimmed down from the original schema after
  `source_batch`/`answerable_from_corpus`/`expected_lectures` were dropped.)
- `L37-41` — `META_PHRASES`: regex patterns matching phrasing that would
  make a question about the lecture series itself rather than about the
  Seerah (e.g. "this lecture", "the shaykh", "Yasir Qadhi", "according to
  the video/series/transcript") — enforcing the question-writer agents'
  brief that questions should sound like something a curious listener would
  ask about history, not about the video series.
- `L43` — `NEAR_DUPLICATE_RATIO = 0.82`: the `SequenceMatcher` similarity
  threshold above which two questions get flagged as probable duplicates.
- `L46-53` — `load_transcripts()`: reads the raw `.jsonl` transcripts,
  returns `{lecture_number: full_text}` — the ground truth for the verbatim
  check.
- `L56-104` — `check_questions(questions, label, transcripts, seen_ids,
  all_questions)` — the core per-question check. `seen_ids` and
  `all_questions` are passed by reference and accumulated **across every
  file processed**; the function returns this batch's own `(problems,
  warnings)`.
  - `L59-64` — computes a display ID (falling back to `"{label}#{index}"`
    if even `question_id` is missing); checks for missing `REQUIRED`
    fields; if any missing, records a problem and skips further checks on
    this malformed question.
  - `L66-68` — checks whether this `question_id` was already seen (across
    *all* files processed so far, not just this one) — flags a duplicate;
    either way records it as now-seen.
  - `L70-72` — checks `tier` is one of `T1`/`T2`/`T3`.
  - `L74-83` — iterates `supporting_quotes`: records the claimed lecture in
    a `quoted` set; flags a problem if that lecture number isn't real;
    flags a problem if the quote text isn't found **verbatim** inside that
    lecture's raw transcript text (the crucial check); flags a
    non-fatal warning if a quote is under 8 words.
  - `L86` — derives the question's actual lecture set from the quotes just
    checked (`lectures = sorted(quoted)`) — the "no stored
    `expected_lectures`" design in action.
  - `L87-88` — T1/T2 must draw quotes from exactly one lecture → problem
    otherwise.
  - `L89-90` — T3 must draw quotes from at least two lectures → problem
    otherwise.
  - `L91-93` — if a T3 question is tagged `cross_episode` and its lecture
    numbers are within 2 of each other, flags a non-fatal warning (doesn't
    force-correct the tag — surfaces it for human judgment).
  - `L94-96` — warns if a T2 question has fewer than 2 supporting quotes.
  - `L98-100` — lowercases the question text, checks it against every
    `META_PHRASES` pattern, flags a problem on the first match.
  - `L102` — appends `(question_id, question_text, batch_label)` to the
    shared `all_questions` list — feeds the cross-batch near-duplicate
    check.
  - `L104` — returns `(problems, warnings)`.
- `L107-114` — `find_near_duplicates(all_questions)`: brute-force all-pairs
  comparison (O(n²) — fine at n=304) using `SequenceMatcher.ratio()` on
  lowercased text; any pair scoring above `NEAR_DUPLICATE_RATIO` is
  recorded with both IDs, their source batches, and a text preview.
- `L117-176` — `main()`:
  - `L118` — resolves the target path — a CLI argument or the default
    `data/eval_questions_raw.json`.
  - `L119-120` — exits with an error if it doesn't exist.
  - `L121` — if the target is a directory, globs `*.json` inside it
    (supports the earlier per-batch-file workflow); otherwise treats it as
    the single file.
  - `L123-127` — loads transcripts once, initializes shared accumulator
    state: `seen_ids` (dict), `all_questions` (list), problem/warning
    lists, tier/cross-episode counters, `lecture_hits` counter.
  - `L129-146` — for each file: loads it, runs `check_questions` (using the
    file's own `"batch"` field or its filename stem as the label), tallies
    tier counts, cross-episode count, and per-lecture hit counts (now
    derived from `supporting_quotes`, matching the design above) into
    running totals; prints a per-file status line; accumulates
    problems/warnings.
  - `L148` — runs the near-duplicate check across everything collected.
  - `L149` — computes the grand total.
  - `L150-158` — prints the overall summary: total and per-tier breakdown,
    the tier split as a percentage triple (compared against the design
    target 20/40/40), and lecture coverage (how many of 104 lectures have
    at least one question, listing any with zero).
  - `L160-168` — for each finding category with entries (PROBLEMS,
    NEAR-DUPLICATES, WARNINGS), prints a header and up to 40 example lines,
    with a "and N more" note if truncated.
  - `L170-172` — exits with code 1 if there were any problems or
    duplicates (so this can gate `make verify`); otherwise prints "No
    problems found."

---

## `seerah/eval/run_retrieval.py` (262 lines)

**Purpose.** Checks retrieval quality against the 304-question set —
before any generation/agent layer exists — in two modes: an interactive
one-question-at-a-time inspector, and a batch scorer computing recall@k
and full_coverage@k per tier. This is the file that produced the earlier
0.687 recall / 0.438 full-coverage numbers.

**Line by line.**
- `L1-33` — docstring: states the two modes; explains the design decision
  to match quotes as **substrings of retrieved chunk text** rather than
  requiring a chunk-index grounding pass first (that pass doesn't exist
  yet); explains why this is a reliable stand-in (chunks came from a
  sentence-aware splitter, quotes were authored as short verbatim spans, so
  "is this quote a substring of this chunk" reliably means "is this the
  right chunk"); notes it runs against the **live** Qdrant/BM25 indexes as
  they exist right now, not a frozen snapshot — so a re-embed or re-chunk
  gets checked automatically the next time this runs; gives usage examples
  for both modes and the available filters.
- `L35-44` — imports: `argparse`, `json`, `re` (whitespace normalization),
  rich UI components, local `config` and the shared `Retriever`.
- `L46` — module-level `console = Console()`.
- `L49-51` — `load_questions()`: reads `eval_questions_raw.json`, returns
  just its `"questions"` list.
- `L54-61` — `filter_questions(questions, tier=None, question_id=None,
  cross_episode=None)`: applies up to three independent filters in
  sequence (exact `question_id`, `tier`, the `cross_episode` boolean) —
  each applied only if its argument isn't `None`/falsy, so any subset of
  filters can be combined.
- `L64-67` — `normalize(text)`: collapses any whitespace run to a single
  space, strips, lowercases — makes the substring match robust to how a
  quote wraps differently once embedded inside a larger chunk (transcripts
  have irregular spacing).
- `L70-77` — `quote_hit_rank(quote, hits)`: normalizes the quote once, then
  walks the hit list in rank order (1-indexed), returning the rank of the
  first hit whose normalized text contains the normalized quote; returns
  `None` if no hit contains it. **The single core primitive** both modes
  build on.
- `L82-95` — `print_summary_table(title, hits, hit_ranks_for_quotes=())`:
  same idea as the plain CLI's table, plus an extra "Quote?" column
  printing bold-green "YES" for any rank present in
  `hit_ranks_for_quotes` — the visual marker for which retrieved chunk(s)
  actually contained a required quote.
- `L98-138` — `run_interactive(questions, retriever, top_k)`:
  - `L100` — prints a ready banner with the loaded question count.
  - `L102-111` — for each question: prints a Panel with the question text
    plus tier/cross_episode metadata, then the reference answer, then every
    supporting quote (truncated to 100 chars) labeled by lecture number.
  - `L113-114` — runs both vector and BM25 search for this question's text.
  - `L116-117` — for each retriever, computes the ranks at which any of
    this question's quotes were found (a walrus operator inside a
    comprehension both computes and filters `quote_hit_rank` in one pass —
    only non-`None` ranks survive).
  - `L120-123` — prints both summary tables, titled with counts like
    "(2/3 quotes found)," passing the found ranks through for highlighting.
  - `L125-131` — prints every retrieved chunk's full text as a Panel,
    appending a green "(quote found)" marker for any matching rank.
  - `L133-137` — prints a separator, then prompts Enter-to-continue/`q`-to-
    quit, catching EOF/interrupt to break cleanly — **this is the loop
    you're currently using** to manually review the zero-hit question
    lists.
- `L143-166` — `score_question(q, retriever, top_k)` — the batch-mode
  per-question scorer:
  - `L144-145` — runs both searches once.
  - `L148-149` — computes the rank (or `None`) for every supporting quote
    against each retriever's hits.
  - `L151-158` — nested `summarize(ranks)`: given one retriever's list of
    ranks/`None`s, computes `quotes_found` (count of non-`None`),
    `quotes_total`, `full_coverage` (bool — found count equals total, i.e.
    *every* required quote was retrieved), `best_rank` (the lowest found
    rank, or `None` — computed here but only surfaced in the per-question
    JSON detail, not the aggregate table).
  - `L160-166` — returns a dict with the question's id/tier/cross_episode
    plus both retrievers' `summarize()` results — exactly the shape
    written into `retrieval_check.json`'s `"per_question"` list, which is
    what the earlier zero-hit dump was built from.
- `L169-192` — `aggregate(results, top_k)`:
  - `L170-172` — groups all results by tier into a dict.
  - `L173-174` — adds two synthetic groups on top of the three real tiers:
    `"ALL"` (every result) and `"cross_episode"` (only cross-episode-tagged
    results) — so the summary table shows both a tier breakdown and these
    two cross-cutting views.
  - `L177-191` — for each group, computes, for both vector and bm25:
    `recall@k` as the mean across that group's questions of
    `quotes_found / quotes_total` (**this is the 0.687/0.520-style
    numbers**), and `full_coverage@k` as the mean of the boolean
    `full_coverage` flag (**this is the 0.438/0.303-style numbers**), both
    rounded to 4 decimals.
  - returns the `summary` dict keyed by tier/group name.
- `L195-225` — `run_batch(questions, retriever, top_k, out_path)`:
  - `L197-200` — scores every question one at a time, printing progress
    every 20 questions (and always on the last one).
  - `L202` — aggregates into the summary dict.
  - `L204-219` — builds and prints a rich table with one row per group
    (T1/T2/T3/cross_episode/ALL, in that fixed order, skipping any group
    absent from the summary when `--tier`/`--cross-episode` filters were
    applied), showing `n` and all four metrics per retriever.
  - `L221-225` — if an output path was given (always true unless
    explicitly disabled), writes JSON with the top_k used, the full summary
    dict, and the complete per-question detail list — exactly
    `data/retrieval_check.json`.
- `L228-262` — `main()`:
  - `L229` — UTF-8 stdout.
  - `L230-233` — argparse with a **mutually exclusive required group** for
    `--interactive` vs `--batch` — exactly one must be passed, enforced by
    argparse itself.
  - `L235-241` — shared flags: `--top-k` (default 10), `--tier` (restricted
    to T1/T2/T3 via `choices`), `--id` (question_id filter, mainly for
    interactive mode), `--cross-episode` (a flag defaulting to `None` so
    "not passed" is distinguishable from "passed as False" — this is why
    `filter_questions` checks `is not None` rather than truthiness),
    `--out` (batch-mode output path override).
  - `L244-249` — loads all questions, applies whichever filters were
    given; if the filtered set is empty, prints an error and returns
    without loading any indexes (avoids the multi-second Qdrant/BM25 load
    cost for a filter that matched nothing).
  - `L251-252` — loads indexes — always both vector and BM25 here (unlike
    the plain CLI, there's no `--retriever` restriction flag in this file).
  - `L254-258` — dispatches to `run_interactive` or `run_batch` based on
    which mode flag was set, defaulting the batch output path to
    `data/retrieval_check.json` if `--out` wasn't given.
  - `L261-262` — entry-point guard.

---

## `Makefile` (144 lines)

**Purpose.** The single command surface for the whole pipeline, applying
the A(rebuild)/B(use committed)/verify pattern consistently across all
four stages, plus setup/query/eval convenience targets.

**Line by line.**
- `L1-13` — header comment: explains the universal three-mode convention
  (default = B = use committed artifact; `-rebuild` = A = actually run;
  `-verify` = check without writing) and notes every target assumes it's
  run from the repo root.
- `L15` — `.DEFAULT_GOAL := help` — bare `make` shows the help text rather
  than doing nothing or erroring.
- `L16-21` — `.PHONY` declares every target name (all of them here) as not
  corresponding to a real output file, so Make never skips a target because
  a same-named file happens to exist.
- `L23` — `PY := python` — the interpreter command defined in exactly one
  place.
- `L27-60` — `help`: a block of `@echo` lines (the leading `@` suppresses
  Make from echoing the command itself) printing a grouped menu — Setup,
  each of the four stages with its cost annotation, then "Use it" — this is
  what bare `make`/`make help` prints.
- `L64-65` — `install`: `pip install -r requirements.txt`.
- `L67-74` — `up`/`down`/`logs`: thin wrappers around `docker compose up
  -d`/`down`/`logs -f qdrant`.
- `L76-79` — `setup`: declared as depending on `up embed text-search`
  (Make's dependency syntax — running `make setup` runs `up`, then `embed`,
  then `text-search`, in that order), then prints a ready message. Comment
  notes the two stages *before* `embed` (chunk, contextualize) are skipped
  because their outputs are already committed to the repository.
- `L83-90` — the three `chunk`/`chunk-rebuild`/`chunk-verify` targets, each
  a one-line call into `seerah.ingest.chunk` with no flag/`--force`/
  `--verify`.
- `L94-101` — the same three-mode pattern for `context`/`context-plan`/
  `context-rebuild` (the "preview" target is named `-plan` here, not
  `-verify`, since `--dry-run` previews a cost estimate rather than diffing
  an existing artifact — a deliberate naming difference reflecting the
  different kind of check).
- `L105-112` — the same pattern for `embed`/`embed-rebuild`/`embed-verify`.
- `L116-120` — the same pattern for `text-search`/`text-search-rebuild`
  (stage 4 has no `-verify` target — there's nothing incremental to verify
  beyond "does the directory exist and have content," which the stage
  itself already checks).
- `L124-125` — `query`: runs `seerah.cli`.
- `L127` — `verify`: depends on `chunk-verify embed-verify
  validate-questions` — runs all three checks in sequence. Note there's no
  contextualize-stage verify in this list — stage 2 has no `--verify` mode
  at all, only `--dry-run`, since "verifying" its output would mean
  re-running the LLM summarization and comparing text, which isn't a
  deterministic operation the way chunking is.
- `L129-130` — `validate-questions`: runs `seerah.eval.validate_questions`.
- `L132-133` — `eval`: runs the **frozen pilot evaluation script directly**
  (`pilot_evaluation/evaluate_retrieval.py`) — notably not part of the
  `seerah` package, since the pilot is explicitly kept as a separate,
  historical experiment rather than live pipeline code.
- `L137-139` — `rebuild-all`: depends on all four stages' rebuild/non-skip
  variants in pipeline order — `chunk-rebuild context embed-rebuild
  text-search-rebuild`. Note it's `context`, not `context-rebuild` — a
  full pipeline rebuild still reuses any already-cached lecture summaries
  rather than discarding them, almost certainly the intended,
  cost-conscious behavior.
- `L141-144` — `clean`: removes the two purely mechanically-rebuildable
  directories (`data/bm25_index`, `data/batch_wave_inputs`) but explicitly
  leaves `data/contextual_cache/` alone, with a comment explaining exactly
  why — those cached summaries cost real money to regenerate, so "clean"
  here means "remove what's free to rebuild," not "remove everything
  generated."

---

## Loose ends worth knowing before building the agent

A few things this walkthrough surfaced that aren't bugs, but are worth
having in mind:

1. **`embed.py`'s embedding cost isn't persisted.** `build()` returns
   `(total_tokens, cost)`, but `main()` discards them (`embed.py:153`).
   Stage 2 records its cost in the artifact (`summary_cost_usd`); stage 3
   currently doesn't, so the $0.28 figure only exists as a printed line,
   not as data.
2. **`Hit.score` is not comparable across retrievers** (`retrieve.py:22`) —
   cosine similarity from Qdrant and a BM25 score are different scales
   entirely. Any future fusion/reranking work needs to normalize or replace
   these scores, not average them directly.
3. **`chunk.py`'s coverage check doesn't hard-fail the stage** — a coverage
   gap only prints a warning (`chunk.py:112-119`), it doesn't raise or
   exit non-zero. Worth tightening if this is ever wired into an automated
   check.
4. **`run_retrieval.py` always loads both indexes** (`run_retrieval.py:252`)
   — unlike `seerah/cli.py`, there's no `--retriever` flag to skip loading
   one of them, so even a `--tier T1` filtered run pays the cost of loading
   both.
