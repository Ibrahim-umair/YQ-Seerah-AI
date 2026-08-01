"""
Generates a labeled question set for retrieval evaluation: for each of the
10 eval-sample lectures, ask an LLM for a few factual questions grounded in
that lecture, each with a VERBATIM supporting quote (not a paraphrase) so a
human can verify the question is actually answerable from the source text.

We do NOT trust the LLM to know chunk boundaries (it never saw the chunking).
Instead, each supporting quote is matched back to the real chunk(s) that
contain it, from recursive_eval_set_results.json's "Recursive (Sentence)"
chunks - that's the ground truth used for scoring retrieval later.

Uses gpt-5.4-mini (not nano) since writing a good, factually-grounded test
question is a harder task than the 1-2 sentence chunk summaries nano handled
fine - a bad question here would quietly corrupt the whole evaluation.

Output: eval_questions.json
"""

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_PATH = REPO_ROOT / "seerah_transcripts.jsonl"
CHUNKS_PATH = REPO_ROOT / "recursive_eval_set_results.json"
OUTPUT_PATH = REPO_ROOT / "eval_questions.json"

TARGET_LECTURES = [8, 10, 21, 34, 44, 53, 66, 76, 89, 100]
QUESTIONS_PER_LECTURE = 3
QUESTION_MODEL = "gpt-5.4-mini"


def load_target_lectures():
    lectures = []
    with open(TRANSCRIPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["lecture_number"] in TARGET_LECTURES:
                lectures.append(record)
    lectures.sort(key=lambda r: TARGET_LECTURES.index(r["lecture_number"]))
    return lectures


def load_chunks_by_lecture():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    by_lecture = {}
    for chunk in data["Recursive (Sentence)"]:
        by_lecture.setdefault(chunk["lecture_number"], []).append(chunk)
    return by_lecture


def generate_questions(lecture):
    prompt = (
        f"This is a lecture titled \"{lecture['canonical_title']}\" from a Seerah "
        f"(life of the Prophet Muhammad) lecture series.\n\n"
        f"<lecture>\n{lecture['text']}\n</lecture>\n\n"
        f"Write exactly {QUESTIONS_PER_LECTURE} factual questions that someone might "
        "genuinely ask about this content, spread across different parts of the "
        "lecture (not all from the introduction). For each question, give:\n"
        "- \"question\": the question itself\n"
        "- \"answer\": a correct, concise answer based on the lecture\n"
        "- \"supporting_quote\": an EXACT, VERBATIM quote of 10-30 words copied "
        "directly from the lecture text above that supports the answer. Do not "
        "paraphrase or summarize the quote - copy it exactly as it appears.\n\n"
        "Return a JSON object: {\"questions\": [{\"question\": ..., \"answer\": ..., "
        "\"supporting_quote\": ...}, ...]}"
    )
    response = client.chat.completions.create(
        model=QUESTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)["questions"]


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def find_matching_chunks(quote, lecture_chunks):
    normalized_quote = normalize(quote)
    matches = []
    for chunk in lecture_chunks:
        if normalized_quote in normalize(chunk["text"]):
            matches.append(chunk["chunk_index"])
    return matches


def main():
    lectures = load_target_lectures()
    chunks_by_lecture = load_chunks_by_lecture()

    all_questions = []
    question_id = 0

    for lecture in lectures:
        print(f"Lecture {lecture['lecture_number']} ({lecture['canonical_title']}): generating questions...")
        questions = generate_questions(lecture)
        lecture_chunks = chunks_by_lecture[lecture["lecture_number"]]

        for q in questions:
            matched_chunks = find_matching_chunks(q["supporting_quote"], lecture_chunks)
            all_questions.append({
                "question_id": question_id,
                "lecture_number": lecture["lecture_number"],
                "canonical_title": lecture["canonical_title"],
                "question": q["question"],
                "answer": q["answer"],
                "supporting_quote": q["supporting_quote"],
                "quote_verified": len(matched_chunks) > 0,
                "matched_chunk_indices": matched_chunks,
            })
            question_id += 1
            status = "OK" if matched_chunks else "!! QUOTE NOT FOUND VERBATIM !!"
            print(f"  [{status}] {q['question']}")

    unverified = [q for q in all_questions if not q["quote_verified"]]
    print(f"\nTotal questions: {len(all_questions)}")
    print(f"Quotes that failed exact-match verification: {len(unverified)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
