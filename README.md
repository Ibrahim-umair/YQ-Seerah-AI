# Seerah Lookup & Summarizer

A RAG (Retrieval-Augmented Generation) application built over Shaykh Dr. Yasir Qadhi's Seerah lecture series — a 104-part lecture course on the life of the Prophet Muhammad ﷺ. This project was built as a capstone for the DataTalks.Club LLM Zoomcamp.

## Problem Statement

Yasir Qadhi's Seerah series is one of the most detailed English-language accounts of the Prophet's life available, but it exists as 104 long-form video lectures (averaging ~13,000 words / ~45-75 minutes each) with no searchable index. If someone wants to know "why did the Quraysh decide to fight at Uhud?" or "what happened during the Prophet's final illness?", their only option today is to scrub through hours of video hoping to land on the right lecture.

This project turns that lecture series into something you can actually query: ask a question in natural language, and get an answer grounded in what Yasir Qadhi actually said, with a citation back to the specific lecture (and its YouTube link) it came from — rather than a generic answer from an LLM's general knowledge, which risks getting specific historical/religious details wrong or ungrounded.

## Dataset

- **Source**: 104 lecture transcripts from the Yasir Qadhi Seerah series, stored as `seerah_transcripts.jsonl` (one JSON object per line).
- **Fields per lecture**: `lecture_number`, `canonical_title`, `youtube_url`, `text` (the full transcript).
- **Scale**: ~1.32 million words / ~1.78 million tokens total across the corpus. Average lecture is ~12,720 words (~17,100 tokens); the longest lecture ("The Death of Prophet Muhammad") is ~18,700 words (~25,300 tokens).
- **Character of the text**: raw spoken-lecture transcript — no paragraph breaks, no timestamps, no headings, and no newline characters anywhere. It's conversational English heavily mixed with transliterated Arabic/Islamic terminology and occasional Quranic Arabic script. This matters a lot for chunking: there is no structural markup to lean on, only the flow of speech itself.

## Data Preparation: Chunking Strategy

Chunks are built with sentence-aware splitting (LlamaIndex `SentenceSplitter`, 800 tokens / 80 token overlap) rather than a fixed-size token window — the splitter packs whole sentences up to the token budget, so a chunk never begins or ends mid-sentence. This matters because the transcripts are raw spoken narrative with no paragraph breaks or headings to lean on otherwise.

---

## Retrieval Evaluation (in progress)

The core evaluation of this project: does adding LLM-generated contextual summaries to chunks (Anthropic's "Contextual Retrieval" technique) actually improve retrieval, measured against real questions with known correct answers.

**Sample set**: 10 lectures (8, 10, 21, 34, 44, 53, 66, 76, 89, 100), chosen to span the series rather than cluster around one narrative arc.

**Setup**:
- Every lecture is chunked twice: `Recursive (Sentence)` (plain chunks) and `Recursive + Contextual` (same chunks, with a short LLM-written summary prepended) — see `scripts/05_recursive_eval_set.py`, output in `recursive_eval_set_results.json`.
- A labeled question set (30 questions, 3 per lecture) was generated from the actual lecture content, each with a verbatim supporting quote traceable back to a real chunk — see `scripts/06_generate_eval_questions.py`, output in `eval_questions.json`. (25 of the 30 have a verified ground-truth chunk; 5 failed automated verbatim-match verification on inspection - false negatives, not bad questions - and were excluded from scoring rather than guessed at.)
- Retrieval is compared across a 2x2 matrix: {plain, contextual} chunks x {vector (BGE-M3 embeddings via Qdrant), BM25 (keyword)} — see `scripts/07_build_vector_index.py`, `scripts/08_build_bm25_index.py`, `docker-compose.yml`.
- Scored with Hit Rate@k and MRR (Mean Reciprocal Rank) against the labeled question set — see `scripts/11_evaluate_retrieval.py`, output in `retrieval_eval_results.json`.

**Reproducing this**: the full pipeline can be re-run end to end (`docker compose up -d` then scripts 05, 07, 08, 11 in order), or the pre-computed results (`recursive_eval_set_results.json`, `eval_questions.json`, `retrieval_eval_results.json`) can be inspected directly without running anything, so this doesn't require an OpenAI API key or a local BGE-M3/Docker setup just to see the data.

### Results (10-lecture pilot, 25 scored questions)

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Vector (plain) | 0.52 | 0.72 | 0.76 | 0.612 |
| Vector (contextual) | 0.40 | 0.84 | 0.96 | 0.618 |
| BM25 (plain) | 0.36 | 0.80 | 0.84 | 0.529 |
| BM25 (contextual) | **0.52** | 0.84 | 0.88 | **0.631** |

**Verdict**: contextual retrieval improves recall for both retrieval methods — Hit@5 and Hit@10 go up across the board when chunks carry a prepended summary, most notably for vector search (0.76 -> 0.96 at Hit@10). Based on this, we're going with contextual retrieval as the approach going forward. One thing we're not glossing over: vector search's Hit@1 actually *drops* with contextual chunks (0.52 -> 0.40) even as its recall further down the list improves - worth investigating further once the eval set grows, rather than assumed away.

**Caveats on this result, stated plainly**: this is a 10-lecture pilot (284 chunks), not the full 104-lecture corpus, and 25 questions is a small enough sample that a couple of questions flipping outcome would move these numbers several points either way. It's also likely that BM25 is doing artificially well here specifically because the evaluation questions were generated by an LLM reading the same transcripts, so it naturally reused the transcripts' exact spelling of names and terms (e.g. "Badr") - a real user has no reason to spell transliterated Arabic terms the way the transcript does (Badr/Badar/Badur, etc.), which would disadvantage keyword-based BM25 more than embedding-based vector search. We plan to address this with LLM-based query rewriting/normalization before retrieval (also satisfies the course's "query rewriting" best-practice criterion), and to re-run this evaluation on a larger, more carefully chunk-grounded question set before treating this as final.
