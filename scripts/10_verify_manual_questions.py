"""
Consolidates the manually-collected per-lecture question files
(manual_qa/lecture_{n}_questions.json, produced by pasting the prompt +
that lecture's numbered chunks into a chat UI) into the final eval_questions.json.

No quote-matching needed this time: the model was given numbered chunks and
told which one to write about, so chunk_index is authoritative by
construction. This script only sanity-checks that the returned chunk_index
actually exists for that lecture (defends against an out-of-range or
hallucinated index).

Output: eval_questions.json
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_PATH = REPO_ROOT / "seerah_transcripts.jsonl"
CHUNKS_PATH = REPO_ROOT / "recursive_eval_set_results.json"
MANUAL_QA_DIR = REPO_ROOT / "manual_qa"
OUTPUT_PATH = REPO_ROOT / "eval_questions.json"

TARGET_LECTURES = [8, 10, 21, 34, 44, 53, 66, 76, 89, 100]


def load_titles_by_lecture():
    titles = {}
    with open(TRANSCRIPTS_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["lecture_number"] in TARGET_LECTURES:
                titles[record["lecture_number"]] = record["canonical_title"]
    return titles


def load_valid_chunk_indices_by_lecture():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    valid = {}
    for chunk in data["Recursive (Sentence)"]:
        valid.setdefault(chunk["lecture_number"], set()).add(chunk["chunk_index"])
    return valid


def main():
    titles = load_titles_by_lecture()
    valid_indices = load_valid_chunk_indices_by_lecture()

    all_questions = []
    question_id = 0
    missing_files = []

    for lecture_number in TARGET_LECTURES:
        path = MANUAL_QA_DIR / f"lecture_{lecture_number}_questions.json"
        if not path.exists():
            missing_files.append(path.name)
            continue

        with open(path, encoding="utf-8") as f:
            questions = json.load(f)

        for q in questions:
            chunk_index = q["chunk_index"]
            valid = chunk_index in valid_indices[lecture_number]
            all_questions.append({
                "question_id": question_id,
                "lecture_number": lecture_number,
                "canonical_title": titles[lecture_number],
                "chunk_index": chunk_index,
                "question": q["question"],
                "answer": q["answer"],
                "chunk_index_valid": valid,
            })
            question_id += 1
            status = "OK" if valid else "!! INVALID CHUNK INDEX !!"
            print(f"  [{status}] (lecture {lecture_number}, chunk {chunk_index}) {q['question']}")

    if missing_files:
        print(f"\nMissing files, skipped: {missing_files}")

    invalid = [q for q in all_questions if not q["chunk_index_valid"]]
    print(f"\nTotal questions: {len(all_questions)}")
    print(f"Questions with an invalid chunk_index: {len(invalid)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
