# Seerah Lookup & Summarizer
#
# Every ingestion stage has the same two modes:
#
#   B (default)  use the committed artifact - free, instant, nothing is recomputed
#   A (-rebuild) actually run the stage from its input
#
# plus a check where one is meaningful:
#
#   -verify      re-derive the work and diff it against the committed artifact,
#                writing nothing
#
# Run every target from the repository root.

.DEFAULT_GOAL := help
.PHONY: help install up up-infra up-proxy down logs setup verify query bot api db \
        chunk chunk-rebuild chunk-verify \
        context context-rebuild context-plan \
        embed embed-rebuild embed-verify \
        text-search text-search-rebuild \
        sentences sentences-clear \
        rebuild-all eval clean

PY := python

# --- help -------------------------------------------------------------------

help:
	@echo "Seerah Lookup & Summarizer"
	@echo ""
	@echo "  Setup"
	@echo "    make install            install pinned dependencies"
	@echo "    make up-infra           start ONLY Qdrant + Postgres (no app/Grafana yet)"
	@echo "    make up / down / logs   start / stop / tail the FULL stack (app, Qdrant, Postgres, Grafana)"
	@echo "    make up-proxy           also start Caddy (real TLS) - needs DOMAIN set in .env, see .env.example"
	@echo "    make db                 create the conversations/feedback Postgres tables"
	@echo "    make setup              everything a fresh clone needs, in the right order:"
	@echo "                            infra up -> embed -> text-search -> db -> full stack up"
	@echo ""
	@echo "  Stage 1 - chunk        transcripts -> data/chunks_plain.json        (free)"
	@echo "    make chunk              use the committed artifact"
	@echo "    make chunk-rebuild      re-chunk from the transcripts"
	@echo "    make chunk-verify       check the artifact still reproduces exactly"
	@echo ""
	@echo "  Stage 2 - context      plain -> data/chunks_contextual.json         (~\$$1.47 full)"
	@echo "    make context            summarize only what is missing or stale"
	@echo "    make context-plan       show what would run and what it costs, submit nothing"
	@echo "    make context-rebuild    discard ALL cached summaries and redo the corpus"
	@echo ""
	@echo "  Stage 3 - embed        contextual -> Qdrant collection              (~\$$0.28)"
	@echo "    make embed              build it if missing, stale or damaged"
	@echo "    make embed-rebuild      delete and rebuild the collection"
	@echo "    make embed-verify       check the live collection against the artifact"
	@echo ""
	@echo "  Stage 4 - text-search  contextual -> data/bm25_index/               (free)"
	@echo "    make text-search        use the existing index"
	@echo "    make text-search-rebuild  rebuild it"
	@echo ""
	@echo "  Stage 3b - sentences   adds/removes per-sentence timestamps on an"
	@echo "                         ALREADY-BUILT collection - no re-embed, free (\$$0)"
	@echo "    make sentences          add them if data/chunks_contextual_with_timestamps.json"
	@echo "                            exists, remove them if it doesn't"
	@echo "    make sentences-clear    force-remove them either way (e.g. to reproduce"
	@echo "                            exactly what a pre-this-feature commit would have)"
	@echo ""
	@echo "  Use it"
	@echo "    make query              interactive raw retrieval, no generation, over all 104 lectures"
	@echo "    make bot                interactive agentic Q&A on the host (add ARGS='--simple' for SeerahRAG)"
	@echo "    make api                run the FastAPI app directly on the host, with --reload"
	@echo "    make verify             run every verification check"
	@echo "    make validate-questions check the evaluation question set's integrity"
	@echo "    make eval               re-run the pilot retrieval evaluation"
	@echo ""
	@echo "    make rebuild-all        rebuild EVERY stage from the transcripts (slow, costs money)"
	@echo ""
	@echo "  Once 'make setup' has run: frontend is 'cd frontend && npm install && npm run dev'."
	@echo "  Grafana dashboard at http://localhost:3000 (admin/admin) - local only, never public."

# --- setup ------------------------------------------------------------------

install:
	$(PY) -m pip install -r requirements.txt

up-infra:
	docker compose up -d qdrant postgres

up:
	docker compose up -d

up-proxy:
	docker compose --profile proxy up -d

down:
	docker compose down

logs:
	docker compose logs -f qdrant

db:
	$(PY) -m seerah.db

bot:
	$(PY) -m seerah.bot $(ARGS)

api:
	$(PY) -m uvicorn seerah.api:app --reload

# Everything a fresh clone needs: the two stages upstream of this are committed.
# Infra (qdrant+postgres) has to come up and get populated BEFORE the full
# stack, or the `app` container crash-loops - it checks the Qdrant collection
# exists at startup, by design (see seerah/retrieve.py).
setup: up-infra embed text-search db up
	@echo ""
	@echo "Ready. Run 'make query' or 'make bot' to search the corpus, or 'make api' to run"
	@echo "the API on the host - the containerized one is already up at http://localhost:8000."

# --- stage 1: chunk ---------------------------------------------------------

chunk:
	$(PY) -m seerah.ingest.chunk

chunk-rebuild:
	$(PY) -m seerah.ingest.chunk --force

chunk-verify:
	$(PY) -m seerah.ingest.chunk --verify

# --- stage 2: contextualize -------------------------------------------------

context:
	$(PY) -m seerah.ingest.contextualize

context-plan:
	$(PY) -m seerah.ingest.contextualize --dry-run

context-rebuild:
	$(PY) -m seerah.ingest.contextualize --force

# --- stage 3: embed ---------------------------------------------------------

embed:
	$(PY) -m seerah.ingest.embed

embed-rebuild:
	$(PY) -m seerah.ingest.embed --force

embed-verify:
	$(PY) -m seerah.ingest.embed --verify

# --- stage 4: BM25 ----------------------------------------------------------

text-search:
	$(PY) -m seerah.ingest.bm25

text-search-rebuild:
	$(PY) -m seerah.ingest.bm25 --force

# --- stage 3b: sentence-level timestamps (reversible, no re-embed) ----------

sentences:
	$(PY) -m seerah.ingest.sentence_timestamps

sentences-clear:
	$(PY) -m seerah.ingest.sentence_timestamps --clear

# --- use it -----------------------------------------------------------------

query:
	$(PY) -m seerah.cli

verify: chunk-verify embed-verify validate-questions

validate-questions:
	$(PY) -m seerah.eval.validate_questions

eval:
	$(PY) pilot_evaluation/evaluate_retrieval.py

# --- full rebuild -----------------------------------------------------------

rebuild-all: chunk-rebuild context embed-rebuild text-search-rebuild
	@echo ""
	@echo "Full rebuild complete."

clean:
	rm -rf data/bm25_index data/batch_wave_inputs
	@echo "Removed the locally rebuildable indexes. Cached summaries in"
	@echo "data/contextual_cache/ were kept - deleting those costs real money to regenerate."
