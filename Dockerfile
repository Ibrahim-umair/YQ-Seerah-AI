# Backend API image. The frontend is deployed separately (static build to
# Vercel or similar) - see README. This image serves the FastAPI app only.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY seerah/ ./seerah/
COPY data/chunks_contextual.json ./data/chunks_contextual.json
COPY data/chunks_contextual_with_timestamps.json ./data/chunks_contextual_with_timestamps.json

# Builds the BM25 index at image-build time (free, local, seconds) from the
# committed chunks_contextual.json - so the image is self-contained and
# doesn't depend on data/bm25_index/ (gitignored) already existing on
# whatever machine runs `docker build`. Picks up sentence-level timestamps
# automatically since chunks_contextual_with_timestamps.json is present -
# see seerah/ingest/bm25.py.
RUN python -m seerah.ingest.bm25

EXPOSE 8000
CMD ["uvicorn", "seerah.api:app", "--host", "0.0.0.0", "--port", "8000"]
