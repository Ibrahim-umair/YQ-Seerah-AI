"""Read/write helpers for the JSON artifacts that connect the pipeline stages.

Each stage writes a self-describing artifact: the chunks themselves plus enough
metadata (strategy, chunk settings, cost) for a reader to know what produced it
without going back to the source.
"""

import json


def write_chunks(path, stage, strategy, chunks, **extra):
    payload = {
        "stage": stage,
        "strategy": strategy,
        "num_chunks": len(chunks),
        "num_lectures": len({c["lecture_number"] for c in chunks}),
        **extra,
        "chunks": chunks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def read_chunks(path):
    """Returns the chunk list. Tolerates the pre-refactor artifact, which stored
    the list under a "Recursive + Contextual" key instead of "chunks"."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the stage that produces it, or pull the "
            f"committed artifact from the repository."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if "chunks" in payload:
        return payload["chunks"]
    for legacy_key in ("Recursive + Contextual", "Recursive (Sentence)"):
        if legacy_key in payload:
            return payload[legacy_key]
    raise ValueError(f"{path} has no recognisable chunk list (keys: {list(payload)})")


def load_sentence_timestamps(path):
    """Returns {(lecture_number, chunk_index): sentences} from the
    with-timestamps chunk artifact, or {} if that file doesn't exist.

    This is an optional enrichment layer on top of the main contextual
    chunks (same text/summary, plus each chunk's transcript sentences
    individually timestamped) - callers (embed.py, bm25.py) should treat a
    missing file as "no sentence-level timing available" and degrade
    gracefully (citation refinement then falls back to chunk-start
    timestamps only), not as an error."""
    if not path.exists():
        return {}
    chunks = read_chunks(path)
    return {(c["lecture_number"], c["chunk_index"]): c.get("sentences", []) for c in chunks}


def load_lectures(transcripts_path):
    lectures = []
    with open(transcripts_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lectures.append(json.loads(line))
    lectures.sort(key=lambda r: r["lecture_number"])
    return lectures


def group_by_lecture(chunks):
    grouped = {}
    for c in chunks:
        grouped.setdefault(c["lecture_number"], []).append(c)
    for lecture_chunks in grouped.values():
        lecture_chunks.sort(key=lambda c: c["chunk_index"])
    return grouped
