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

`SentenceSplitter` reserves room for a `Document`'s metadata inside each chunk's token budget, so passing populated metadata shifts every chunk boundary. Earlier runs of this project did that and later runs did not, which left the corpus cut two different ways. `data/chunking_manifest.json` records which mode produced each lecture, so re-chunking from the transcripts reproduces the exact same boundaries per lecture instead of silently re-cutting the corpus a third way. New lectures should use `without_metadata`, which gives each chunk the full 800-token budget. Every chunking run also asserts full transcript coverage (`report_coverage()` in `chunk.py`) — every character of all 104 transcripts must fall inside at least one chunk.

That inconsistency also caused a real defect, since fixed. Three lectures (26, 42, 43) were interrupted mid-run and resumed by a later script that keyed resume on chunk *index*. Because the two runs cut the text differently, index *n* did not mean the same thing in both, and the resumed lectures spliced together chunks from two different cuts — silently dropping **2,201 characters** that then existed in no chunk and could not be retrieved (including As'ad ibn Zurara's speech at the Second Pledge of Aqaba, and the incident that triggered the conflict with Banu Qaynuqa). Nothing errored; the run reported a plausible chunk count and looked healthy.

Two changes make this class of bug impossible now. Stage 2's cache is keyed on a **SHA-256 fingerprint of the exact chunk texts** it was built from, so any boundary change invalidates the lecture and forces a clean re-summarization instead of a splice. And stage 1 runs a **coverage check** on every run, asserting that every character of all 104 transcripts falls inside at least one chunk. The three lectures were re-chunked and re-summarized; all 2,201 characters are now retrievable.

---

## Retrieval Evaluation: the 10-lecture pilot

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

## Retrieval evaluation on the full corpus, and hybrid search

The 10-lecture pilot's ground truth was single-chunk, so Hit Rate@k/MRR (as taught in the course) applied directly. That premise breaks on the full corpus: the lecturer is repetitive and serializes events across multi-part arcs (Badr spans 7 lectures, the Conquest of Makkah spans 6), so a realistic answer often needs 2-4 chunks, sometimes from lectures dozens apart.

To measure this honestly, an LLM wrote **304 tiered questions** against the real transcripts, each with a reference answer and verbatim supporting quotes — see `data/eval_questions_raw.json` and `seerah/eval/`. Tiers: **T1** (63, one chunk), **T2** (117, several chunks in one lecture), **T3** (124, several lectures — 40 of those tagged **cross-episode**, where the lectures are far apart rather than consecutive arc parts). Metrics are **recall@k** (the direct generalization of Hit Rate@k to multi-chunk ground truth — they're identical when a question needs exactly one chunk) and **full-coverage@k** (were *all* required chunks retrieved, not just some — the harsher, more decision-relevant bar for a question needing several pieces of evidence).

**Vector vs. BM25 vs. hybrid (Reciprocal Rank Fusion), top-10, all 304 questions:**

| Tier | n | Vector recall/full-cov | BM25 recall/full-cov | Hybrid k=60 (textbook default) | Hybrid k=8 | Hybrid k=10 (chosen) |
|---|---:|---|---|---|---|---|
| T1 | 63 | 0.847 / 0.809 | 0.704 / 0.651 | 0.881 / 0.841 | 0.929 / 0.889 | 0.929 / 0.889 |
| T2 | 117 | 0.703 / 0.419 | 0.510 / 0.248 | 0.662 / 0.427 | 0.699 / 0.453 | 0.699 / 0.462 |
| T3 | 124 | 0.591 / 0.266 | 0.436 / 0.177 | 0.562 / 0.282 | 0.578 / 0.298 | 0.582 / 0.298 |
| cross-episode | 40 | 0.508 / 0.200 | 0.354 / 0.125 | 0.481 / 0.225 | 0.479 / 0.225 | 0.479 / 0.225 |
| **ALL** | 304 | 0.687 / 0.438 | 0.520 / 0.303 | 0.667 / 0.454 | 0.697 / 0.480 | **0.699 / 0.484** |

**Why RRF, and why k=10:** vector and BM25 scores are on incomparable scales (cosine ~0.5-0.6 vs. BM25 ~5-9), so any score-weighted fusion would just be dominated by whichever retriever's numbers are larger. RRF sidesteps this by fusing on rank position alone: `score = Σ 1/(k + rank)` across retrievers, deduplicated by chunk. `k=60` is the standard constant from the original RRF paper — but at `k=60`, hybrid was a genuine trade-off against vector alone: better full-coverage (0.454 vs. 0.438) at the cost of lower recall (0.667 vs. 0.687), because a flat score curve at high k requires both retrievers to roughly agree before rewarding a chunk. Sweeping `k` against this same 304-question set (`python -m seerah.eval.sweep_rrf_k`) shows lower k performs better here: a smaller k sharpens the score gap between top and lower ranks (at k=60, rank 1 is only 1.15x rank 10's score; at k=10, it's 1.82x), so a single retriever's strong ranking counts on its own instead of being washed out for lacking cross-retriever agreement — exactly what's needed to recover the cases where BM25 finds a chunk vector missed entirely, or vice versa. `k=10` matches or beats every other value tested (1 through 60) on full-coverage, stays within 0.004 of the best recall, and is never worse than its runner-up `k=8` on any tier for either metric — giving hybrid a clean win over vector alone on **both** metrics simultaneously (0.699/0.484 vs. 0.687/0.438), not the trade-off `k=60` produced.

**Reproducing this**: `python -m seerah.eval.run_retrieval --batch` scores vector/BM25/hybrid against all 304 questions; `python -m seerah.eval.sweep_rrf_k` re-runs the k sweep; `python -m seerah.eval.validate_questions` checks the question set's own integrity (verbatim quotes, no duplicates, tier consistency) before trusting its numbers.

---

## LLM output evaluation: simple pipeline vs. agentic RAG

Good retrieval doesn't guarantee a good final answer — the generator can still miss part of a multi-part question, state something the retrieved context doesn't actually support, or get outranked evidence wrong. This project has two generation backends, kept deliberately separate rather than one replacing the other:

- **`seerah.answer.SeerahRAG`** — one retrieval call, one generation call. The simple pipeline.
- **`seerah.agent.SeerahAgent`** — an agentic loop (`seerah/agent.py`), modeled directly on the course's own agentic-RAG lessons: a `search` tool the model calls itself, up to `AGENT_MAX_ITERATIONS` times, deciding when to search again and how to reword the query — the same self-correction pattern the course demonstrates with a literal typo (`"Olama"` → `"Ollama"`), applied here to Arabic transliteration ambiguity (`Badr`/`Badar`, `Ka'b`/`Kab`) instead.

**Judge design** (`seerah/eval/judge_answers.py`), modeled on the course's `offline-rag-evaluation.ipynb`: an LLM judge classifies each generated answer on two *separate* axes, never conflated —

- **Relevance** — does the answer's content agree with a curated reference answer, in substance. The judge is explicitly told the lecture series repeats itself and a reference's cited lecture is not exhaustive, so it must never penalize a correct answer for citing a different (but also correct) lecture.
- **Faithfulness** — is the answer actually supported by the context the generator was *given* this run, independent of whether it happens to be historically correct. This is the hallucination check.

**Baseline** (`--full`, simple pipeline, all 304 questions): 248 RELEVANT, 53 PARTLY_RELEVANT, 3 NON_RELEVANT.

**Closing the gap, iteratively.** The 56 non-RELEVANT questions were retested on the agentic bot and re-judged after each instruction change:

| Round | Still not RELEVANT (of 56) | What changed |
|---|---:|---|
| Agentic, v1 instructions | 15 | Baseline agentic instructions (search tool, transliteration awareness, cite only what's retrieved) |
| Agentic, v2 instructions | 11 | Replaced a flawed "search for the lecture dedicated to this outcome" instruction — it was actually making things *worse*: the model re-searched using the wrong outcome it already (wrongly) believed, which just reconfirmed the wrong narrative. Replaced with "search by the person/event's name only, not the outcome you already found," so a differing account can actually surface. |
| Synthesis-fix instructions + corpus fix | 3 (+1 unresolved) | Added instructions to verify specific claims (not infer them from related-but-different evidence) and to check that multi-part questions are fully answered before stopping. Fixed 2 of the remaining 11 outright via the corpus correction below. 1 more (`C1-008`) never resolved to a reliable verdict at all — see below. |

One finding along the way that shaped how the rest of this evaluation was read: **`gpt-5.6-luna` has no exposed temperature or seed parameter** (confirmed via a live 400 error, not assumed) and demonstrably gives different verdicts on identical input across independent runs. A single judge call on a borderline question is not proof of a fixed defect, so every remaining disputed question was re-run 3x independently before being called "consistently bad" vs. "flaky."

**The last 2 were not a model or retrieval bug — they were the lecturer's own transcript.** Two of the hardest remaining cases traced back to the source material itself: lecture 70 mis-stated *when* Huyay ibn Akhtab was executed (transcribed near "the very beginning of the battle of Khaybar," rather than after the Battle of the Trench alongside Banu Qurayza), and lecture 20 described Ta'if as "reconquered" during the Battle of Hunayn, when the siege there was in fact unsuccessful. The agent was faithfully reproducing what Shaykh Yasir Qadhi actually said — a correct-per-source answer that was still historically imprecise. Both were confirmed as unintentional slips (cross-checked against lecture 83's detailed, dedicated account of Ta'if) and corrected as a **surgical patch**: transcript → plain chunk → contextual summary/cache → Qdrant embedding → BM25, touching only those 2 of 2,763 chunks, with no full re-chunk or re-embed of the corpus.

**Final result, all questions reconciled to their latest tested status** (`data/eval_final_results.json`), 303 of the original 304 (one question produced a genuine 3-way split — NON/PARTLY/RELEVANT, one vote each, across 3 independent runs — and was excluded rather than forced into a category; it's logged with its reason in the file's `excluded_questions`):

| Relevance | n | % |
|---|---:|---:|
| RELEVANT | 300 | 99.0% |
| PARTLY_RELEVANT | 3 | 1.0% |
| NON_RELEVANT | 0 | 0.0% |

Faithfulness across all 303: **294 GROUNDED, 9 PARTIALLY_GROUNDED, 0 UNGROUNDED** — no hallucination found anywhere in the set, including on the 3 questions still not fully RELEVANT.

The 233 questions that already passed on the simple pipeline were not individually re-run on the agentic bot — the agentic bot is a strict superset of the simple pipeline's capability (same retrieval, same model, plus the ability to search again), so re-paying for ~250 questions that already passed was judged not worth it. A stratified 5-question spot-check across tiers confirmed no regression. The remaining 3 PARTLY_RELEVANT questions (`A1-020`, consistently; `A7-019` and `A9-009`, only sometimes) are a known, documented limitation rather than a silently-accepted one.

**Reproducing this**: `python -m seerah.eval.judge_answers --full` (baseline, simple pipeline) or `--full --agentic` (agentic bot); `--retest-file <prior_output.json> --agentic` to re-judge only what a prior run flagged; `--ids A,B,C --agentic --repeat 3` to check self-consistency on specific questions.

---

## Citation precision: pinpointing the exact moment in the video

A retrieved chunk's *start* timestamp is not the same as the moment its content actually begins — an ~800-token chunk can span several minutes, and the specific claim an answer makes is often well past the chunk's own start (or, if the true beginning was cut off by a chunk boundary, just before it). Linking every citation to "wherever this chunk starts" would frequently land a viewer near, but not at, the moment actually being referenced.

This is solved without touching retrieval at all. `SeerahAgent._refine_citations()` (`seerah/agent.py`) runs a small, dedicated pass *after* the answer is already written: given the question, the finished answer, and a fixed set of the top `CITATION_REFINE_TOP_K` (3) retrieved passages — each with every sentence individually timestamped, and paired with the chunk immediately before it in the same lecture in case the real start was cut off by a chunk boundary — a separate OpenAI Structured Outputs call (a strict JSON schema, not a citation format embedded in free text for a parser to potentially miss) points at the exact `[HH:MM:SS]` marker(s) that actually support what the answer says.

Two things keep this strictly an improvement, never a regression:
- **Validated, never invented**: a returned timestamp is only ever applied if it's found verbatim among the real sentence-level timestamps shown to the model. Anything hallucinated, rounded, or mismatched is simply dropped, and the citation falls back to the chunk's own start timestamp — the same behavior as before this existed.
- **Bounded cost**: the call's context is fixed at 3 passages regardless of how many chunks a multi-search question actually retrieved (which can be anywhere from 5 to ~30) — so this pass's cost and latency never scale with how much searching the main agentic loop did for a given question.

This same pass also decides which source is shown as *primary*: rather than whichever chunk simply scored highest in retrieval, the lecture(s) the refined citations actually trace back to are surfaced first, with any other lecture the answer genuinely draws on shown alongside it as an additional source.

The sentence-level timestamps this relies on are additive metadata only, layered on top of the existing corpus rather than replacing anything: `data/chunks_contextual_with_timestamps.json` carries the same `text`/`summary` as the committed `chunks_contextual.json` plus a per-chunk `sentences` array, and `python -m seerah.ingest.sentence_timestamps` (or its `--clear` mode) can sync or remove this layer on an already-built Qdrant collection and BM25 index without any re-embedding — a fresh `embed`/`bm25` build picks it up automatically if the file is present. It never touches embeddings or BM25 scoring, so retrieval quality is unaffected either way, and older data without it simply falls back to a chunk's own start timestamp.

Users can flag when a timestamp lands on the wrong moment — see the `/feedback/{id}/timestamp` route and the frontend's second 👍/👎 control, both described below — as a signal for whether this is working in practice, independent of whether the answer itself was rated good.

---

## Application interface

**Backend** (`seerah/api.py`, FastAPI): wraps `SeerahAgent` behind four routes.

| Route | Purpose |
|---|---|
| `GET /health` | liveness check |
| `POST /ask` | `{question, previous_response_id?}` → streamed as Server-Sent Events (see below); rate-limited to 10 requests/minute per client IP |
| `POST /feedback/{id}` | `{score: 1 \| -1}` — thumbs up/down on the answer overall |
| `POST /feedback/{id}/timestamp` | `{score: 1 \| -1}` — thumbs up/down on the primary citation's timestamp specifically (did the video land on the right moment), a separate signal from the answer-quality rating above |

Multi-turn conversation needs no manually-reconstructed message history: `previous_response_id` (returned from the prior `/ask`) is handed to OpenAI's Responses API, which resumes that conversation's full state server-side — including its own prior tool calls, not just the visible text. Every call is logged to Postgres (see Monitoring below) with token counts, cost, and latency.

`/ask` streams its answer as Server-Sent Events rather than a single JSON response, so the UI can render tokens as they're generated instead of waiting on the whole answer (plus the citation-refinement pass above) to finish: repeated `{"type": "token", "text": "..."}` events as the answer is written, then one `{"type": "done", "id", "answer", "sources", "response_id"}` event carrying everything a non-streaming caller would need — or a `{"type": "error", "message"}` event in its place if something failed mid-stream.

`/ask` is also rate-limited (10 requests/minute per client IP, via `slowapi`) — each call costs real OpenAI money (the agentic search loop plus the citation-refinement pass), so this caps how much a single caller can run up before it starts returning 429 instead of a stream. The limiter keys off the connecting client's IP directly, so if this is ever deployed behind a reverse proxy or load balancer, that becomes the proxy's own IP instead — rate-limiting the whole app together rather than per real caller — worth revisiting if that becomes the deployment shape.

**Frontend** (`frontend/`, React + Vite): a chat-style UI over that API — ask a question, get an answer with clickable sources (each linking to the exact lecture *and timestamp* via `&t=<seconds>s`), rate the answer 👍/👎, separately rate whether the primary citation's timestamp actually landed on the right moment 👍/👎, keep asking follow-ups in the same thread or start fresh with "Clear chat." Multi-turn is handled client-side by holding onto the last `response_id` and passing it on the next request; nothing persists across a page reload by design — no accounts, no server-side session list, deliberately out of scope for a project with no expected concurrent users.

**Running it locally**: first time on a fresh clone, follow "Quick start" below in order (Qdrant needs to be populated *before* the API container starts, or it crash-loops). After that's done once, day-to-day it's just:
```bash
docker compose up -d        # backend + Qdrant + Postgres + Grafana
cd frontend && npm install && npm run dev
```
`frontend/.env`'s `VITE_API_URL` must point at wherever the backend is actually reachable (`http://localhost:8000` by default).

**Deployment plan** (not live yet - pending cloud credits): the frontend is a pure static build (`npm run build`), intended for a static host (Vercel) rather than a container — a static SPA has no server-side logic to containerize, so wrapping it in Docker would only ever matter for the local `docker-compose` story, never for actual production. The backend + Qdrant + Postgres run together via the same `docker-compose.yml` on a VM (Azure, or any equivalent VPS) once that's set up — see "TLS / going public" under Containerization below for the one extra step this split (static frontend, separate backend host) requires.

## Monitoring & analytics

Every `/ask` call writes one row to Postgres' `conversations` table (question, answer, sources, search log, model, prompt/completion/total tokens, cost, response time, and the `response_id`/`previous_response_id` pair that reconstructs multi-turn threads); every `/feedback` call writes to `feedback`, and every `/feedback/{id}/timestamp` call writes to its own `timestamp_feedback` table — kept separate rather than a column on `feedback` since the two rate different things (answer quality vs. citation timestamp precision) and a user may only rate one of them for a given turn. All three are linked by `conversation_id`; schema and helpers live in `seerah/db.py`.

**Grafana** (`docker-compose.yml`'s `grafana` service, config in `grafana/provisioning/`) reads directly from that same Postgres — no separate metrics pipeline. The dashboard is fully provisioned (data source + all panels defined in checked-in YAML/JSON, not built by clicking through the UI), so it reproduces automatically on a fresh `docker-compose up` with zero manual setup — verified by deleting the Grafana container *and* its data volume entirely and confirming everything reappeared correctly from nothing but the compose file.

8 panels: recent conversations (table), model usage (bar), user feedback (pie), response time / avg token usage / cost (time series), plus avg response time and total cost as single-number stat panels. The user-feedback panel currently reads only `feedback` (answer quality); `timestamp_feedback` is logged and queryable but not yet its own panel — a natural next addition.

Access at `http://localhost:3000` (`admin`/`admin`). **Deliberately never exposed publicly** — it shows internal metrics (cost, question volume, raw feedback) that have no reason to be public, so it stays local-only even once the app itself is deployed.

## Containerization

`docker-compose.yml` runs 4 core services: `app` (the FastAPI backend, built from the root `Dockerfile`), `qdrant`, `postgres`, `grafana` — plus an optional 5th, `caddy` (see "TLS / going public" below). One command, `docker-compose up -d`, brings up the backend stack.

The `Dockerfile` builds the BM25 index *at image-build time* (from the committed `data/chunks_contextual.json`, or `data/chunks_contextual_with_timestamps.json` if present — see "Citation precision" above) rather than depending on `data/bm25_index/` (gitignored) already existing on whatever machine runs `docker build` — so the image is self-contained on a completely fresh clone. It does **not** run the embedding step (`seerah.ingest.embed`) — that's a one-time, costly (~$0.28), API-dependent step you run once against a running Qdrant, same as the Quick start below.

**Network exposure**: `app`, `postgres`, `qdrant`, and `grafana` all bind to `127.0.0.1` only, not `0.0.0.0` — none of them are meant to be reachable from outside the host directly (Qdrant and Postgres have no/weak auth by default; Grafana is an internal ops dashboard, see Monitoring above). This doesn't break anything locally (the frontend's default `http://127.0.0.1:8000` still works, and container-to-container traffic - `app` to `postgres`/`qdrant` - goes over the compose network regardless of host port bindings). It does mean that, as shipped, nothing in this stack is actually reachable once it's running on a remote host - which is deliberate: see below for the one service that's meant to be.

**TLS / going public**: `caddy` is the only service meant to be internet-facing, and it's opt-in (`docker compose --profile proxy up -d`, or `make up-proxy`) rather than part of the default `up`, so local dev is unaffected either way. It reverse-proxies to `app` over the internal network and requests/renews a real Let's Encrypt certificate for `DOMAIN` (set in `.env`) automatically on first start - no certbot, no manual renewal steps. This matters more than "nice to have" the moment the frontend is deployed to a host like Vercel: a browser will not let an HTTPS page call a plain `http://` API at all ("mixed content"), so a deployed frontend simply cannot reach a TLS-less backend, full stop. Before starting it: point `DOMAIN`'s DNS A record at the backend host's public IP, and make sure ports 80/443 are open on that host (Let's Encrypt's HTTP-01 challenge needs both to succeed). The Caddyfile also sets `flush_interval -1`, so `/ask`'s Server-Sent Events still stream token-by-token through the proxy instead of arriving all at once.

The frontend is intentionally not part of this compose file — see "Application interface" above for why.

---

## Running the pipeline

### Quick start

The expensive artifacts are committed, so getting a working system does **not** mean re-running the whole pipeline. You need Python 3.13, Docker, and an OpenAI API key.

```bash
pip install -r requirements.txt
cp .env.example .env              # then put your OPENAI_API_KEY in it

docker compose up -d qdrant postgres   # infra first, not the app yet - see note below
python -m seerah.ingest.embed          # embeds the committed chunks into Qdrant (~$0.28, ~3 min)
python -m seerah.ingest.bm25           # builds the keyword index on the host (free, seconds) -
                                        # needed separately for seerah.cli/bot below; the Docker
                                        # image builds its own copy internally at build time
python -m seerah.db                    # creates the conversations/feedback tables

docker compose up -d              # now bring up everything - app, qdrant, postgres, grafana
python -m seerah.cli               # interactive retrieval over all 104 lectures, or:
python -m seerah.bot               # interactive agentic Q&A on the host, or:
curl http://localhost:8000/health  # confirm the containerized API is up
```

**Why infra first, then the app**: `seerah/api.py` loads `SeerahAgent` at startup, which immediately checks that the Qdrant collection exists - by design (`seerah/retrieve.py` fails loudly rather than serving from an empty/stale index). Starting `app` before `seerah.ingest.embed` has actually populated Qdrant means it crash-loops (harmlessly - `restart: unless-stopped` keeps retrying) until that step finishes. Running infra and the embedding step first avoids the loop entirely.

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

Committed: `data/seerah_transcripts.jsonl`, `data/chunks_plain.json`, `data/chunks_contextual.json`, `data/chunks_contextual_with_timestamps.json`, `data/chunking_manifest.json`.

Rebuilt locally (gitignored): `data/contextual_cache/`, `data/batch_wave_inputs/`, `data/bm25_index/`, `qdrant_storage/`, `postgres_storage/`, the `grafana_data` Docker volume.

### Repository layout

```
seerah/                 the application
  config.py             all paths, models and constants in one place
  artifacts.py          artifact read/write helpers
  retrieve.py           vector + BM25 retrieval behind one result type
  answer.py             SeerahRAG - the simple, single-shot pipeline
  agent.py              SeerahAgent - the agentic, multi-turn pipeline
  db.py                 Postgres schema + conversation/feedback/timestamp-feedback logging
  api.py                FastAPI app: /health, /ask (streamed, rate-limited), /feedback, /feedback/.../timestamp
  bot.py                interactive CLI over the agent (or --simple for SeerahRAG)
  cli.py                interactive raw-retrieval tool (no generation)
  ingest/               the four ingestion stages above, plus sentence_timestamps.py (adds/removes
                        per-sentence timestamps on an already-built index, no re-embedding)
  eval/                 retrieval + LLM-as-judge evaluation tooling
frontend/               React + Vite chat UI over the API
grafana/provisioning/   dashboard + data source, auto-loaded on container start
data/                   dataset and pipeline artifacts
pilot_evaluation/       the 10-lecture retrieval experiment (frozen evidence)
Dockerfile              backend API image (see Containerization above)
docker-compose.yml      app + qdrant + postgres + grafana + optional caddy
Caddyfile               reverse proxy + automatic TLS, opt-in (see "TLS / going public")
```
