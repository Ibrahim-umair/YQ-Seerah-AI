# Seerah Lookup & Summarizer

A RAG (Retrieval-Augmented Generation) application built over Shaykh Dr. Yasir Qadhi's Seerah lecture series — a 104-part lecture course on the life of the Prophet Muhammad ﷺ. This project was built as a capstone for the DataTalks.Club LLM Zoomcamp.

## Problem Statement

Yasir Qadhi's Seerah series is one of the most detailed English-language accounts of the Prophet's life available, but it exists as 104 long-form video lectures (averaging ~13,000 words / ~45-75 minutes each) with no searchable index. If someone wants to know "why did the Quraysh decide to fight at Uhud?" or "what happened during the Prophet's final illness?", their only option today is to scrub through hours of video hoping to land on the right lecture.

This project turns that lecture series into something you can actually query: ask a question in natural language, and get an answer grounded in what Yasir Qadhi actually said, with a citation back to the specific lecture (and its YouTube link) it came from — rather than a generic answer from an LLM's general knowledge, which risks getting specific historical/religious details wrong or ungrounded.

## Dataset

- **Source**: 104 lecture transcripts from the Yasir Qadhi Seerah series, stored as `data/seerah_transcripts.jsonl` (one JSON object per line).
- **Fields per lecture**: `lecture_number`, `canonical_title`, `youtube_url`, `text` (the full transcript).
- **Scale**: ~1.32 million words / ~1.78 million tokens total across the corpus. Average lecture is ~12,720 words (~17,100 tokens); the longest lecture ("The Death of Prophet Muhammad") is ~18,700 words (~25,300 tokens).
- **Character of the text**: raw spoken-lecture transcript — no paragraph breaks, no timestamps, no headings, and no newline characters anywhere. It's conversational English heavily mixed with transliterated Arabic/Islamic terminology and occasional Quranic Arabic script. This matters a lot for chunking: there is no structural markup to lean on, only the flow of speech itself.

## Data Preparation: Chunking Strategy

Chunks are built with sentence-aware splitting (LlamaIndex `SentenceSplitter`, 800 tokens / 80 token overlap) rather than a fixed-size token window — the splitter packs whole sentences up to the token budget, so a chunk never begins or ends mid-sentence. This matters because the transcripts are raw spoken narrative with no paragraph breaks or headings to lean on otherwise.

---

### Chunking reproducibility, and a data defect this uncovered

`SentenceSplitter` reserves room for a `Document`'s metadata inside each chunk's token budget, so passing populated metadata shifts every chunk boundary. Earlier runs of this project did that and later runs did not, which left the corpus cut two different ways. `data/chunking_manifest.json` records which mode produced each lecture, so `python -m seerah.ingest.chunk --verify` reproduces the committed artifact byte-for-byte. New lectures should use `without_metadata`, which gives each chunk the full 800-token budget.

That inconsistency also caused a real defect, since fixed. Three lectures (26, 42, 43) were interrupted mid-run and resumed by a later script that keyed resume on chunk *index*. Because the two runs cut the text differently, index *n* did not mean the same thing in both, and the resumed lectures spliced together chunks from two different cuts — silently dropping **2,201 characters** that then existed in no chunk and could not be retrieved (including As'ad ibn Zurara's speech at the Second Pledge of Aqaba, and the incident that triggered the conflict with Banu Qaynuqa). Nothing errored; the run reported a plausible chunk count and looked healthy.

Two changes make this class of bug impossible now. Stage 2's cache is keyed on a **SHA-256 fingerprint of the exact chunk texts** it was built from, so any boundary change invalidates the lecture and forces a clean re-summarization instead of a splice. And stage 1 runs a **coverage check** on every run, asserting that every character of all 104 transcripts falls inside at least one chunk. The three lectures were re-chunked and re-summarized; all 2,201 characters are now retrievable.

---

## Retrieval Evaluation (in progress)

The core evaluation of this project: does adding LLM-generated contextual summaries to chunks (Anthropic's "Contextual Retrieval" technique) actually improve retrieval, measured against real questions with known correct answers.

**Sample set**: 10 lectures (8, 10, 21, 34, 44, 53, 66, 76, 89, 100), chosen to span the series rather than cluster around one narrative arc.

**Setup** (everything for this pilot lives in `pilot_evaluation/`):
- Every lecture is chunked twice: `Recursive (Sentence)` (plain chunks) and `Recursive + Contextual` (same chunks, with a short LLM-written summary prepended) — see `build_pilot_chunks.py`, output in `recursive_eval_set_results.json`.
- A labeled question set (30 questions, 3 per lecture) was generated from the actual lecture content, each with a verbatim supporting quote traceable back to a real chunk — see `generate_eval_questions.py`, output in `eval_questions.json`. (25 of the 30 have a verified ground-truth chunk; 5 failed automated verbatim-match verification on inspection - false negatives, not bad questions - and were excluded from scoring rather than guessed at.)
- Retrieval is compared across a 2x2 matrix: {plain, contextual} chunks x {vector (embeddings via Qdrant), BM25 (keyword)} — see `build_bm25_index.py`, `build_openai_index.py`, `docker-compose.yml`. (The BGE-M3 index builder that produced the first table below has since been removed — see "Why BGE-M3 was dropped".)
- Scored with Hit Rate@k and MRR (Mean Reciprocal Rank) against the labeled question set — see `evaluate_retrieval.py`, output in `retrieval_eval_results.json`.

**Reproducing this**: the pilot can be re-run end to end (`docker compose up -d`, then `build_pilot_chunks.py`, `build_bm25_index.py`, `build_openai_index.py`, `evaluate_retrieval.py` in that order — or just `make eval` for the scoring step). Or the pre-computed results (`recursive_eval_set_results.json`, `eval_questions.json`, `retrieval_eval_results.json`) can be inspected directly without running anything, so this doesn't require an OpenAI API key or a Docker setup just to see the data.

This pilot is a frozen historical experiment — it is kept as the evidence behind the design decisions, and is deliberately not wired into the production pipeline described under "Running the pipeline" at the end of this document.

### Initial exploration: BGE-M3 vs. BM25 (10-lecture pilot, 25 scored questions)

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Vector (BGE-M3, plain) | 0.52 | 0.72 | 0.76 | 0.612 |
| Vector (BGE-M3, contextual) | 0.40 | 0.84 | 0.96 | 0.618 |
| BM25 (plain) | 0.36 | 0.80 | 0.84 | 0.529 |
| BM25 (contextual) | 0.52 | 0.84 | 0.88 | 0.631 |

**Verdict on chunking**: contextual retrieval improves recall for both methods here — Hit@5 and Hit@10 go up across the board when chunks carry a prepended summary, most notably for vector search (0.76 -> 0.96 at Hit@10). One thing we're not glossing over: vector search's Hit@1 actually *drops* with contextual chunks (0.52 -> 0.40) even as its recall further down the list improves.

**Why BGE-M3 was dropped** (its numbers above stay as-is, kept as evidence - not deleted): BGE-M3 runs locally on CPU here (no GPU available), and per-query embedding alone measured ~400-500ms - heavy for a model that, per the results above, wasn't even the strongest option. Once OpenAI's `text-embedding-3-large` was tried (below) and clearly outperformed it, there was no reason to keep spending further evaluation effort on BGE-M3 specifically. Iterating away from a weaker embedder once a stronger, cheaper-to-query option is found is expected practice, not a shortcut - this is also called out directly in the course FAQ as a normal part of the process.

The BGE-M3 **code** has since been removed as well (`build_bge_index.py`, and the BGE half of `evaluate_retrieval.py`). Keeping it would have meant carrying `sentence-transformers` and a multi-GB `torch` install in the dependency list purely to reproduce a result that had already lost — a real cost to anyone setting the project up, for no decision-making value. What is kept is the **evidence**: the measurements above, the reasoning here, and the `vector_plain` / `vector_contextual` rows inside `retrieval_eval_results.json`. Re-running the evaluation carries those rows forward automatically and labels them `archived`, so dropping the code cannot quietly erase the numbers this README reports.

### Ongoing evaluation matrix: BM25 vs. OpenAI `text-embedding-3-large` (same pilot, same 25 questions)

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 (plain) | 0.36 | 0.80 | 0.84 | 0.529 |
| BM25 (contextual) | 0.52 | 0.84 | 0.88 | 0.631 |
| Vector (OpenAI large, plain) | 0.48 | 0.84 | 0.92 | 0.633 |
| Vector (OpenAI large, contextual) | **0.56** | **1.00** | **1.00** | **0.726** |

*(See `pilot_evaluation/build_openai_index.py` and `pilot_evaluation/evaluate_retrieval.py` - same 284 chunks per variant, same 25-question set, embedded with `text-embedding-3-large` (3072-dim) instead of the local BGE-M3 model. Every retriever condition is scored in a single pass through the same ranking and metric functions, so the rows are directly comparable rather than produced by separate implementations that could drift.)*

**Verdict**: OpenAI large + contextual chunks wins outright - best on every metric, including a **perfect Hit@5 and Hit@10** (the correct chunk was found in the top 5, and thus top 10, for all 25 questions with no exceptions). Contextual retrieval helps here too, and now on a *second* embedding model, not just BGE-M3 - plain scores 0.48/0.84/0.92/0.633, contextual jumps to 0.56/1.00/1.00/0.726. That consistency across two different embedding models is stronger evidence for contextual retrieval than either result alone. The real tradeoff isn't cost (a ~10-word query costs a negligible fraction of a cent to embed) - it's that this path requires a live call to OpenAI's API per query, versus BGE-M3 running fully offline with no external dependency. Whether that tradeoff is worth it depends on whether the deployed app can assume reliable internet access to OpenAI at query time.

**Caveats on all of this, stated plainly**: this is a 10-lecture pilot (284 chunks), not the full 104-lecture corpus, and 25 questions is a small enough sample that a couple of questions flipping outcome would move these numbers several points either way. It's also likely that BM25 is doing artificially well here specifically because the evaluation questions were generated by an LLM reading the same transcripts, so it naturally reused the transcripts' exact spelling of names and terms (e.g. "Badr") - a real user has no reason to spell transliterated Arabic terms the way the transcript does (Badr/Badar/Badur, etc.), which would disadvantage keyword-based BM25 more than embedding-based vector search. We plan to address this with LLM-based query rewriting/normalization before retrieval (also satisfies the course's "query rewriting" best-practice criterion), and to re-run this evaluation on a larger, more carefully chunk-grounded question set before treating this as final.

---

## Scaling to the full corpus

With the retrieval approach settled — sentence-aware chunks, contextual summaries, `text-embedding-3-large` — the same pipeline was run over all 104 lectures, producing **2,763 contextual chunks**.

Doing this synchronously does not work: each contextual-summary request carries the full ~17k-token lecture as context, so the account's 200,000 TPM synchronous limit is hit constantly. The pipeline uses OpenAI's **Batch API** instead, which draws on a separate quota and is 50% cheaper. Batch has its own constraint though — a 2,000,000 enqueued-token cap, against a corpus that totals ~26M enqueued tokens. Submitting everything at once is rejected outright with `token_limit_exceeded`. So whole lectures are bin-packed by their real token count into sequential waves, each under budget, submitted one at a time and polled to completion before the next.

Total cost for the full corpus: **$1.47** for contextual summaries (batch pricing, ~80% prompt-cache hit rate) plus **$0.28** for embeddings.

---

## Running the pipeline

### Quick start

The expensive artifacts are committed, so getting a working system does **not** mean re-running the whole pipeline. You need Python 3.13, Docker, and an OpenAI API key.

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your OPENAI_API_KEY in it
docker compose up -d          # starts Qdrant on localhost:6333

python -m seerah.ingest.embed # embeds the committed chunks into Qdrant (~$0.28, ~3 min)
python -m seerah.ingest.bm25  # builds the keyword index (free, seconds)

python -m seerah.cli          # interactive retrieval over all 104 lectures
```

With `make` installed, that whole sequence is `make install && make setup && make query`. Run `make help` to list every target.

### The four ingestion stages

Ingestion is split so each stage can be run, inspected and re-run independently. Every stage reads the previous stage's committed artifact, and **skips its work if its own output already exists** — so the two stages that cost money are ones you should never need to run.

| Stage | Command | Input → output | Cost |
|---|---|---|---|
| 1. Chunk | `python -m seerah.ingest.chunk` | transcripts → `data/chunks_plain.json` | free, ~40s |
| 2. Contextualize | `python -m seerah.ingest.contextualize` | plain → `data/chunks_contextual.json` | ~$1.47, hours |
| 3. Embed | `python -m seerah.ingest.embed` | contextual → Qdrant collection | ~$0.28, ~3 min |
| 4. BM25 | `python -m seerah.ingest.bm25` | contextual → `data/bm25_index/` | free, seconds |

Each stage offers the same two modes, and a check where one is meaningful:

| | Flag | `make` target | What it does |
|---|---|---|---|
| **Use the committed data** | *(default)* | `make chunk` | Skips the work entirely — free and instant |
| **Actually run it** | `--force` | `make chunk-rebuild` | Rebuilds from the stage's input |
| **Check it** | `--verify` | `make chunk-verify` | Re-derives the work, diffs it against the artifact, writes nothing |

Stage 2 also has `--dry-run` (`make context-plan`), which reports exactly which lectures would be submitted and what it would cost before spending anything.

### What's committed vs. rebuilt locally

Committed: `data/seerah_transcripts.jsonl`, `data/chunks_plain.json`, `data/chunks_contextual.json`, `data/chunking_manifest.json`.

Rebuilt locally (gitignored): `data/contextual_cache/`, `data/batch_wave_inputs/`, `data/bm25_index/`, `qdrant_storage/`.

### Repository layout

```
seerah/                 the application
  config.py             all paths, models and constants in one place
  artifacts.py          artifact read/write helpers
  retrieve.py           vector + BM25 retrieval behind one result type
  cli.py                interactive query tool
  ingest/               the four stages above
data/                   dataset and pipeline artifacts
pilot_evaluation/       the 10-lecture retrieval experiment (frozen evidence)
dropped_manual_question_workflow/   an approach that was tried and abandoned
```
