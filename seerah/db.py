"""Conversation/feedback logging - the persistence layer behind the app
interface's monitoring. Modeled on the LLM Zoomcamp course's own
05-monitoring/code/db_init.py + db_save.py + db_feedback.py: raw SQL via
psycopg (no ORM), two tables linked by a foreign key.

    conversations - one row per question answered: the question, the answer,
                    the sources/search log it was grounded in, token counts,
                    cost, and how long it took.
    feedback      - one row per rating a user gives an answer (+1/-1),
                    pointing back at the conversation it rates.

Usage:
    python -m seerah.db            # create both tables if they don't exist
    python -m seerah.db --drop     # drop and recreate both (schema changes)
"""

import argparse

import psycopg
from psycopg.types.json import Jsonb

from seerah import config


def get_db_connection():
    return psycopg.connect(
        host=config.POSTGRES_HOST,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
    )


def init_db(drop=False):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS feedback")
                cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources JSONB NOT NULL,
                    search_log JSONB,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost FLOAT NOT NULL,
                    response_time FLOAT NOT NULL,
                    response_id TEXT,
                    previous_response_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                    score INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_conversation(question, answer, sources, search_log, model,
                      prompt_tokens, completion_tokens, total_tokens, cost, response_time,
                      response_id=None, previous_response_id=None):
    """Inserts one answered question. Returns the new row's id, which the
    caller (the API) hands back to the client so a later feedback call can
    reference this exact conversation.

    response_id/previous_response_id record the OpenAI conversation chain for
    this turn (see seerah.agent.SeerahAgent.ask) - not required for logging to
    work, but needed to reconstruct which turns belong to the same
    multi-turn conversation later."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, sources, search_log, model,
                    prompt_tokens, completion_tokens, total_tokens, cost, response_time,
                    response_id, previous_response_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (question, answer, Jsonb(sources), Jsonb(search_log), model,
                 prompt_tokens, completion_tokens, total_tokens, cost, response_time,
                 response_id, previous_response_id),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
        return conversation_id
    finally:
        conn.close()


def save_feedback(conversation_id, score):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, score) VALUES (%s, %s)",
                (conversation_id, score),
            )
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--drop", action="store_true", help="drop both tables first (schema changes)")
    args = parser.parse_args()

    init_db(drop=args.drop)
    print("Database ready: conversations, feedback")


if __name__ == "__main__":
    main()
