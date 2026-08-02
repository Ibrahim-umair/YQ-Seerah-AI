"""Stage 1 - split the 104 lecture transcripts into sentence-aligned chunks.

Free and local: no API calls, runs in well under a minute.

Chunking uses LlamaIndex's SentenceSplitter (800 tokens / 80 overlap) rather
than a fixed-size token window, so a chunk never begins or ends mid-sentence.
The transcripts are raw spoken narrative with no paragraph breaks or headings
to lean on, so sentence boundaries are the only structure available.

    Input:  data/seerah_transcripts.jsonl, data/chunking_manifest.json
    Output: data/chunks_plain.json

About the manifest: SentenceSplitter reserves room for a Document's metadata
inside each chunk's token budget, so passing populated metadata shifts every
boundary. Earlier runs of this project did that; the later batch run did not.
The manifest records which mode produced each lecture, so this stage
reproduces the committed artifact byte-for-byte instead of silently re-cutting
the corpus. Lectures added from here on should use `without_metadata`, which
gives each chunk the full 800-token budget.

Usage:
    python -m seerah.ingest.chunk           # skips if the output already exists
    python -m seerah.ingest.chunk --force   # re-chunk from the transcripts
    python -m seerah.ingest.chunk --verify  # re-chunk and diff against the artifact
"""

import argparse
import json

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from seerah import artifacts, config

STRATEGY = "Recursive (Sentence)"


def load_manifest():
    with open(config.MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["lectures"]


def chunk_lecture(lecture, splitter, mode):
    """mode: 'with_metadata' or 'without_metadata' - see the module docstring."""
    metadata = {}
    if mode == "with_metadata":
        metadata = {
            "lecture_number": lecture["lecture_number"],
            "canonical_title": lecture["canonical_title"],
            "youtube_url": lecture["youtube_url"],
        }
    doc = Document(text=lecture["text"], metadata=metadata)
    nodes = splitter.get_nodes_from_documents([doc])

    return [
        {
            "strategy": STRATEGY,
            "chunk_index": i,
            "lecture_number": lecture["lecture_number"],
            "canonical_title": lecture["canonical_title"],
            "youtube_url": lecture["youtube_url"],
            "text": node.get_content(),
            "word_count": config.count_words(node.get_content()),
            "token_count": config.count_tokens(node.get_content()),
        }
        for i, node in enumerate(nodes)
    ]


def build_chunks():
    lectures = artifacts.load_lectures(config.TRANSCRIPTS_PATH)
    manifest = load_manifest()
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    chunks = []
    for lecture in lectures:
        mode = manifest.get(str(lecture["lecture_number"]), "without_metadata")
        chunks.extend(chunk_lecture(lecture, splitter, mode))
    return lectures, chunks


def report_coverage(lectures, chunks):
    """Every character of every transcript should live in at least one chunk.
    This is the check that would have caught the lecture 26/42/43 gaps, where a
    resumed run spliced chunks cut two different ways and silently dropped the
    text in between."""
    by_lecture = artifacts.group_by_lecture(chunks)
    total_missing = 0

    for lecture in lectures:
        n = lecture["lecture_number"]
        text = lecture["text"]
        spans = []
        for c in by_lecture.get(n, []):
            start = text.find(c["text"])
            if start < 0:
                print(f"  WARNING: lecture {n} chunk {c['chunk_index']} is not verbatim in the transcript")
                continue
            spans.append((start, start + len(c["text"])))
        spans.sort()

        missing = spans[0][0] if spans else len(text)
        for i in range(1, len(spans)):
            gap = spans[i][0] - spans[i - 1][1]
            if gap > 1:  # a 1-char gap is just the whitespace between sentences
                missing += gap
        tail = len(text) - max((e for _, e in spans), default=0)
        if tail > 1:
            missing += tail

        if missing > 0:
            print(f"  WARNING: lecture {n} has {missing} chars covered by no chunk")
            total_missing += missing

    if total_missing == 0:
        print("  coverage check: every character of all 104 transcripts is inside a chunk")
    else:
        print(f"  coverage check: {total_missing} chars are in NO chunk - investigate before continuing")
    return total_missing


def verify_against_artifact(chunks):
    existing = artifacts.read_chunks(config.PLAIN_CHUNKS_PATH)
    if len(existing) != len(chunks):
        print(f"MISMATCH: artifact has {len(existing)} chunks, re-chunking produced {len(chunks)}")
        return False
    differing = [
        (c["lecture_number"], c["chunk_index"])
        for c, e in zip(chunks, existing)
        if c["text"] != e["text"]
    ]
    if differing:
        print(f"MISMATCH: {len(differing)} chunks differ, first at {differing[0]}")
        return False
    print(f"VERIFIED: re-chunking reproduces all {len(chunks)} chunks in the artifact exactly")
    return True


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="re-chunk even if the artifact exists")
    parser.add_argument("--verify", action="store_true", help="re-chunk and diff against the artifact, writing nothing")
    args = parser.parse_args()

    if args.verify:
        _, chunks = build_chunks()
        raise SystemExit(0 if verify_against_artifact(chunks) else 1)

    if config.PLAIN_CHUNKS_PATH.exists() and not args.force:
        existing = artifacts.read_chunks(config.PLAIN_CHUNKS_PATH)
        print(f"{config.PLAIN_CHUNKS_PATH.name} already exists ({len(existing)} chunks) - using it as is.")
        print("Pass --force to re-chunk from the transcripts, or --verify to check it still reproduces.")
        return

    print(f"Chunking {config.TRANSCRIPTS_PATH.name} at {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} tokens...")
    lectures, chunks = build_chunks()
    report_coverage(lectures, chunks)

    artifacts.write_chunks(
        config.PLAIN_CHUNKS_PATH,
        stage="chunk",
        strategy=STRATEGY,
        chunks=chunks,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    print(f"Wrote {len(chunks)} chunks from {len(lectures)} lectures -> {config.PLAIN_CHUNKS_PATH}")


if __name__ == "__main__":
    main()
