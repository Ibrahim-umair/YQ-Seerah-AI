"""Evaluation tooling.

    python -m seerah.eval.validate_questions        # check the committed question set

The question set itself is data/eval_questions_raw.json - 304 tiered questions
written against the full 104-lecture corpus. "raw" means pre-grounding: each
question carries CANDIDATE supporting quotes proposed when it was written, which
an independent grounding pass still has to confirm and resolve to chunk sets
before they can be used as retrieval labels.
"""
