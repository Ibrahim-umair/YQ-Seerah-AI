"""
Exports each of the 10 eval-sample lectures as its NUMBERED CHUNKS (the same
"Recursive (Sentence)" chunks already sitting in recursive_eval_set_results.json)
plus a reusable prompt template, so you can manually paste each lecture into
a chat UI and get back questions that are directly tied to a real chunk_index
- no quote-matching or reverse-engineering needed, because the model is told
exactly which chunk to write about and says so directly in its answer.

Output: manual_qa/lecture_{n}_chunks.txt for each lecture, manual_qa/PROMPT_TEMPLATE.txt

Workflow:
  1. Run this script once.
  2. For each lecture_{n}_chunks.txt: paste PROMPT_TEMPLATE.txt + that file's
     contents into the chat UI.
  3. Save the raw JSON array the model returns as manual_qa/lecture_{n}_questions.json
  4. Once all 10 are done, run scripts/10_verify_manual_questions.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "recursive_eval_set_results.json"
OUTPUT_DIR = REPO_ROOT / "manual_qa"

TARGET_LECTURES = [8, 10, 21, 34, 44, 53, 66, 76, 89, 100]
QUESTIONS_PER_LECTURE = 5

PROMPT_TEMPLATE = f"""You are helping build an evaluation set for a RAG system. Below are the NUMBERED CHUNKS of one lecture from a Seerah (life of the Prophet Muhammad) lecture series by Yasir Qadhi. Each [N] marks the start of chunk N.

<chunks>
PASTE THE LECTURE'S CHUNKS HERE
</chunks>

Write exactly {QUESTIONS_PER_LECTURE} factual questions, each grounded in a SINGLE specific chunk (do not combine information from two different chunks into one question). Spread your chosen chunks out across the lecture - don't pick 5 chunks that are all next to each other.

For each question, provide:
1. "chunk_index": the number of the chunk this question is grounded in - must be one of the [N] labels above, and the answer must be fully supported by that chunk's own text.
2. "question": the question itself, phrased naturally.
3. "answer": a correct, concise answer based on that chunk.

Return ONLY a JSON array, no other text before or after it:
[
  {{"chunk_index": 0, "question": "...", "answer": "..."}},
  ...
]
"""


def load_chunks_by_lecture():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    by_lecture = {}
    for chunk in data["Recursive (Sentence)"]:
        by_lecture.setdefault(chunk["lecture_number"], []).append(chunk)
    for chunks in by_lecture.values():
        chunks.sort(key=lambda c: c["chunk_index"])
    return by_lecture


def format_numbered_chunks(chunks):
    return "\n\n".join(f"[{c['chunk_index']}] {c['text']}" for c in chunks)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    chunks_by_lecture = load_chunks_by_lecture()

    (OUTPUT_DIR / "PROMPT_TEMPLATE.txt").write_text(PROMPT_TEMPLATE, encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'PROMPT_TEMPLATE.txt'}")

    for lecture_number in TARGET_LECTURES:
        chunks = chunks_by_lecture[lecture_number]
        title = chunks[0]["canonical_title"]
        out_path = OUTPUT_DIR / f"lecture_{lecture_number}_chunks.txt"
        header = f"Title: {title}\n\n"
        out_path.write_text(header + format_numbered_chunks(chunks), encoding="utf-8")
        print(f"Wrote {out_path} ({len(chunks)} chunks)")

    print(f"\nDone. For each lecture_{{n}}_chunks.txt: paste PROMPT_TEMPLATE.txt, then paste "
          f"that file's contents where it says 'PASTE THE LECTURE'S CHUNKS HERE'.")
    print("Save each response as manual_qa/lecture_{n}_questions.json, then run "
          "scripts/10_verify_manual_questions.py")


if __name__ == "__main__":
    main()
