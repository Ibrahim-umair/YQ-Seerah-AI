"""FastAPI wrapper around the agentic bot, with conversation/feedback logging
to Postgres and multi-turn support.

Usage:
    uvicorn seerah.api:app --reload

Then, with the server running:
  - Open http://127.0.0.1:8000/docs for FastAPI's interactive test page
    (Swagger UI) - it lets you fill in and send requests from the browser,
    no Postman needed, though Postman works identically against the same URL.
  - GET  http://127.0.0.1:8000/health              -> quick "is it up" check
  - POST http://127.0.0.1:8000/ask                 -> {"question": "..."}
  - POST http://127.0.0.1:8000/feedback/{id}       -> {"score": 1 | -1}, rates the answer
  - POST http://127.0.0.1:8000/feedback/{id}/timestamp -> {"score": 1 | -1}, rates the
    primary citation's timestamp precision specifically (separate signal, see seerah.db)

Multi-turn: every /ask response includes "response_id". To continue that
conversation on the next question, send it back as "previous_response_id" in
the next /ask request. Omit it to start a fresh, unrelated conversation.

/ask streams its answer as Server-Sent Events (not a single JSON body) -
Swagger UI's "Try it out" will send the request fine but only shows the raw
event text, not a rendered stream. Events, in order:
  data: {"type": "status", "text": "..."}     - zero or more, before any "token" -
    the agent's own one-sentence reason for a search it's running, never the
    raw search query (see seerah.agent.SeerahAgent.ask_stream)
  data: {"type": "token", "text": "..."}      - repeated, as the answer is written
  data: {"type": "done", "id": ..., "answer": ..., "sources": [...], "response_id": "..."}
  data: {"type": "error", "message": "..."}   - instead of "done", if something failed

/ask is rate-limited to 10 requests/minute per client IP (real OpenAI cost
per call) - past that, it returns 429 with a plain-text "Rate limit exceeded"
body instead of an SSE stream.
"""

import json
from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from seerah import db
from seerah.agent import SeerahAgent

app = FastAPI(title="Seerah RAG API")

# The React dev server (localhost:5173) and the API (localhost:8000) are
# different origins as far as the browser is concerned, so without this the
# browser blocks every request before it even reaches these routes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# /ask costs real OpenAI money per call (the search loop plus the citation
# refinement pass) - keyed by client IP so one caller can't run up the bill
# for everyone. get_remote_address reads request.client.host directly, which
# is the real caller's IP for a direct connection - if this ever sits behind
# a reverse proxy/load balancer, that will instead be the proxy's own IP
# (rate-limiting the whole app together), so revisit this if that changes.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Loaded once when the server starts, not per-request - the agent holds the
# Qdrant client and the BM25 index in memory, both expensive to reload
# (same reasoning as seerah.bot and seerah.cli).
agent = SeerahAgent()


class AskRequest(BaseModel):
    # max_length blocks context-stuffing (a huge pasted question inflating a
    # single call's cost) - 2000 chars comfortably fits genuine multi-part
    # questions while rejecting anything pathological. FastAPI returns a 422
    # automatically if this is violated.
    question: str = Field(min_length=1, max_length=2000)
    previous_response_id: str | None = None


class FeedbackRequest(BaseModel):
    score: Literal[1, -1]


@app.get("/health")
def health():
    return {"status": "ok"}


def sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/ask")
@limiter.limit("10/minute")
def ask(request: Request, body: AskRequest):
    def event_stream():
        try:
            for event in agent.ask_stream(body.question, previous_response_id=body.previous_response_id):
                if event["type"] == "status":
                    yield sse({"type": "status", "text": event["text"]})
                    continue

                if event["type"] == "token":
                    yield sse({"type": "token", "text": event["text"]})
                    continue

                # event["type"] == "done" - the only other kind ask_stream yields
                # hits already arrive in display order (agent.py's citation
                # refinement puts what the answer actually cites first, then
                # the rest by score) - re-sorting by score here would undo that.
                sources_data = [
                    {"lecture_number": h.lecture_number, "canonical_title": h.canonical_title,
                     "youtube_url": h.youtube_url, "chunk_index": h.chunk_index, "score": h.score,
                     "start_timestamp": h.start_timestamp, "start_timestamp_seconds": h.start_timestamp_seconds,
                     "citation": h.citation, "timestamped_url": h.timestamped_url}
                    for h in event["hits"]
                ]
                usage = event["usage"]
                conversation_id = db.save_conversation(
                    question=body.question, answer=event["answer"], sources=sources_data,
                    search_log=event["search_log"], model=usage["model"],
                    prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"], cost=usage["cost"],
                    response_time=usage["response_time"], citation_time=usage["citation_time"],
                    response_id=usage["response_id"],
                    previous_response_id=body.previous_response_id,
                )
                yield sse({"type": "done", "id": conversation_id, "answer": event["answer"],
                          "sources": sources_data, "response_id": usage["response_id"]})
        except Exception as exc:
            yield sse({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/feedback/{conversation_id}")
def feedback(conversation_id: int, request: FeedbackRequest):
    try:
        db.save_feedback(conversation_id, request.score)
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status_code=404, detail=f"No conversation with id {conversation_id}")
    return {"status": "ok"}


@app.post("/feedback/{conversation_id}/timestamp")
def timestamp_feedback(conversation_id: int, request: FeedbackRequest):
    """Rates the primary citation's timestamp precision specifically - a
    separate signal from /feedback's answer-quality rating (see seerah.db)."""
    try:
        db.save_timestamp_feedback(conversation_id, request.score)
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status_code=404, detail=f"No conversation with id {conversation_id}")
    return {"status": "ok"}
