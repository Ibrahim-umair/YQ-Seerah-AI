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
.PHONY: help install up down logs setup verify query \
        chunk chunk-rebuild chunk-verify \
        context context-rebuild context-plan \
        embed embed-rebuild embed-verify \
        text-search text-search-rebuild \
        rebuild-all eval clean

PY := python

# --- help -------------------------------------------------------------------

help:
	@echo "Seerah Lookup & Summarizer"
	@echo ""
	@echo "  Setup"
	@echo "    make install            install pinned dependencies"
	@echo "    make up / down / logs   start / stop / tail Qdrant"
	@echo "    make setup              up + embed + text-search  (everything a fresh clone needs)"
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
	@echo "  Use it"
	@echo "    make query              interactive retrieval over all 104 lectures"
	@echo "    make verify             run every verification check"
	@echo "    make eval               re-run the pilot retrieval evaluation"
	@echo ""
	@echo "    make rebuild-all        rebuild EVERY stage from the transcripts (slow, costs money)"

# --- setup ------------------------------------------------------------------

install:
	$(PY) -m pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f qdrant

# Everything a fresh clone needs: the two stages upstream of this are committed.
setup: up embed text-search
	@echo ""
	@echo "Ready. Run 'make query' to search the corpus."

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

# --- use it -----------------------------------------------------------------

query:
	$(PY) -m seerah.cli

verify: chunk-verify embed-verify

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
