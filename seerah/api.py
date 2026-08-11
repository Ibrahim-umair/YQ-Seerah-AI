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
  - POST http://127.0.0.1:8000/feedback/{id}       -> {"score": 1 | -1}

Multi-turn: every /ask response includes "response_id". To continue that
conversation on the next question, send it back as "previous_response_id" in
the next /ask request. Omit it to start a fresh, unrelated conversation.
"""

from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Loaded once when the server starts, not per-request - the agent holds the
# Qdrant client and the BM25 index in memory, both expensive to reload
# (same reasoning as seerah.bot and seerah.cli).
agent = SeerahAgent()


class AskRequest(BaseModel):
    question: str
    previous_response_id: str | None = None


class Source(BaseModel):
    lecture_number: int
    canonical_title: str
    youtube_url: str
    chunk_index: int
    score: float
    start_timestamp: str
    start_timestamp_seconds: float
    citation: str
    timestamped_url: str


class AskResponse(BaseModel):
    id: int
    answer: str
    sources: list[Source]
    response_id: str


class FeedbackRequest(BaseModel):
    score: Literal[1, -1]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    answer, hits, search_log, usage = agent.ask(
        request.question, previous_response_id=request.previous_response_id)
    sources_data = [
        {"lecture_number": h.lecture_number, "canonical_title": h.canonical_title,
         "youtube_url": h.youtube_url, "chunk_index": h.chunk_index, "score": h.score,
         "start_timestamp": h.start_timestamp, "start_timestamp_seconds": h.start_timestamp_seconds,
         "citation": h.citation, "timestamped_url": h.timestamped_url}
        for h in sorted(hits, key=lambda h: h.score, reverse=True)
    ]
    conversation_id = db.save_conversation(
        question=request.question, answer=answer, sources=sources_data, search_log=search_log,
        model=usage["model"], prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"], total_tokens=usage["total_tokens"],
        cost=usage["cost"], response_time=usage["response_time"],
        response_id=usage["response_id"], previous_response_id=request.previous_response_id,
    )
    return AskResponse(id=conversation_id, answer=answer,
                       sources=[Source(**s) for s in sources_data],
                       response_id=usage["response_id"])


@app.post("/feedback/{conversation_id}")
def feedback(conversation_id: int, request: FeedbackRequest):
    try:
        db.save_feedback(conversation_id, request.score)
    except psycopg.errors.ForeignKeyViolation:
        raise HTTPException(status_code=404, detail=f"No conversation with id {conversation_id}")
    return {"status": "ok"}
