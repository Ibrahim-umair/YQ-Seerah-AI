"""Integrity check for the evaluation question set.

    python -m seerah.eval.validate_questions              # the committed set
    python -m seerah.eval.validate_questions <path>       # a specific file or directory

Checks that are worth re-running any time the set changes:

  - schema: every question carries the required fields
  - unique question_ids, and no near-duplicate question wording across batches
  - tier consistency: T1/T2 draw on exactly one lecture, T3 on two or more
  - cross_episode questions genuinely span distant lectures, not adjacent ones
  - every supporting quote appears CHARACTER-FOR-CHARACTER in its lecture
  - no meta-questions about the lecture series rather than about the Seerah

The verbatim quote check is the important one. Quotes are the only thread back
from a question to the corpus, and a quote that has been paraphrased or tidied
cannot be located again - it silently becomes an unlabelled question.

There is no separate expected_lectures field: the set of lectures a question
draws on is derived here from the lecture_number values in its
supporting_quotes, since the two were always identical in practice and keeping
both risked them silently drifting apart.
"""

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from seerah import config

REQUIRED = {"question_id", "tier", "cross_episode", "question", "reference_answer",
            "supporting_quotes"}

META_PHRASES = [
    r"\bthis lecture\b", r"\bthe lecture\b", r"\bthe shaykh\b", r"\bthe speaker\b",
    r"\byasir qadhi\b", r"\baccording to the (video|series|transcript)\b",
    r"\bin (this|the) (video|series|episode)\b",
]

NEAR_DUPLICATE_RATIO = 0.82


def load_transcripts():
    texts = {}
    with open(config.TRANSCRIPTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                texts[record["lecture_number"]] = record["text"]
    return texts


def check_questions(questions, label, transcripts, seen_ids, all_questions):
    problems, warnings = [], []

    for i, q in enumerate(questions):
        qid = q.get("question_id", f"{label}#{i}")
        missing = REQUIRED - set(q)
        if missing:
            problems.append(f"{qid}: missing fields {sorted(missing)}")
            continue

        if qid in seen_ids:
            problems.append(f"{qid}: duplicate question_id (also in {seen_ids[qid]})")
        seen_ids[qid] = label

        tier = q["tier"]
        if tier not in ("T1", "T2", "T3"):
            problems.append(f"{qid}: bad tier {tier!r}")

        quoted = set()
        for sq in q["supporting_quotes"]:
            n, quote = sq.get("lecture_number"), sq.get("quote", "")
            quoted.add(n)
            if n not in transcripts:
                problems.append(f"{qid}: quote cites unknown lecture {n}")
            elif quote not in transcripts[n]:
                problems.append(f"{qid}: quote NOT verbatim in lecture {n}: {quote[:70]!r}")
            if len(quote.split()) < 8:
                warnings.append(f"{qid}: very short quote ({len(quote.split())} words)")

        # the set of lectures a question draws on = the lectures its quotes cite
        lectures = sorted(quoted)
        if tier in ("T1", "T2") and len(lectures) != 1:
            problems.append(f"{qid}: {tier} quotes must all come from 1 lecture, got {lectures}")
        if tier == "T3" and len(lectures) < 2:
            problems.append(f"{qid}: T3 must have quotes from 2+ lectures, got {lectures}")
        if tier == "T3" and q["cross_episode"] and len(lectures) >= 2:
            if max(lectures) - min(lectures) <= 2:
                warnings.append(f"{qid}: tagged cross_episode but lectures {lectures} are adjacent")
        if tier == "T2" and len(q["supporting_quotes"]) < 2:
            warnings.append(f"{qid}: T2 should need 2+ passages, only "
                            f"{len(q['supporting_quotes'])} quote(s)")

        text = q["question"].lower()
        if any(re.search(p, text) for p in META_PHRASES):
            problems.append(f"{qid}: meta question about the lectures - {q['question'][:70]!r}")

        all_questions.append((qid, q["question"], label))

    return problems, warnings


def find_near_duplicates(all_questions):
    dupes = []
    for i in range(len(all_questions)):
        for j in range(i + 1, len(all_questions)):
            a, b = all_questions[i], all_questions[j]
            if SequenceMatcher(None, a[1].lower(), b[1].lower()).ratio() > NEAR_DUPLICATE_RATIO:
                dupes.append(f"{a[0]} ({a[2]}) ~ {b[0]} ({b[2]}): {a[1][:60]!r}")
    return dupes


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DATA_DIR / "eval_questions_raw.json"
    if not target.exists():
        raise SystemExit(f"{target} not found")
    files = sorted(target.glob("*.json")) if target.is_dir() else [target]

    transcripts = load_transcripts()
    seen_ids, all_questions = {}, []
    all_problems, all_warnings = [], []
    tiers, cross = Counter(), 0
    lecture_hits = Counter()

    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", [])
        problems, warnings = check_questions(
            questions, data.get("batch", path.stem), transcripts, seen_ids, all_questions
        )
        for q in questions:
            tiers[q.get("tier")] += 1
            cross += bool(q.get("cross_episode"))
            for sq in q.get("supporting_quotes", []):
                lecture_hits[sq.get("lecture_number")] += 1
        status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
        print(f"{path.name:26s} {len(questions):4d} questions  {status}"
              + (f", {len(warnings)} warning(s)" if warnings else ""))
        all_problems += problems
        all_warnings += warnings

    dupes = find_near_duplicates(all_questions)
    total = sum(tiers.values())

    print(f"\n{'=' * 74}")
    print(f"TOTAL {total} questions   T1={tiers['T1']}  T2={tiers['T2']}  T3={tiers['T3']}"
          f"   cross_episode={cross}")
    if total:
        print(f"tier split: {tiers['T1'] * 100 // total}/{tiers['T2'] * 100 // total}"
              f"/{tiers['T3'] * 100 // total}   (design target 20/40/40)")
    uncovered = [n for n in range(1, 105) if n not in lecture_hits]
    print(f"lectures covered: {len(lecture_hits)}/104"
          + (f"   NOT covered: {uncovered}" if uncovered else ""))

    for heading, items in (("PROBLEMS", all_problems),
                           ("NEAR-DUPLICATES", dupes),
                           ("WARNINGS", all_warnings)):
        if items:
            print(f"\n--- {heading} ({len(items)}) ---")
            for line in items[:40]:
                print(f"  {line}")
            if len(items) > 40:
                print(f"  ... and {len(items) - 40} more")

    if all_problems or dupes:
        raise SystemExit(1)
    print("\nNo problems found.")


if __name__ == "__main__":
    main()
