"""
The actual retrieval evaluation: embeds eval_questions.json's questions with
BGE-M3 and scores 4 retrievers against them:

    - recursive_plain       (vector / BGE-M3)
    - recursive_contextual  (vector / BGE-M3)
    - recursive_plain       (BM25)
    - recursive_contextual  (BM25)

A "hit" means the correct chunk (per matched_chunk_indices in
eval_questions.json) appeared in a retriever's top-k results, for the same
lecture_number. Questions with no verified ground truth (empty
matched_chunk_indices - from the quote-matching false negatives we already
diagnosed) are skipped, not counted as misses, since we don't actually know
what a "correct" answer looks like for them.

Metrics: Hit Rate @1 / @5 / @10, and MRR (Mean Reciprocal Rank).

Output: retrieval_eval_results.json
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from llama_index.retrievers.bm25 import BM25Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "eval_questions.json"
BM25_DIR = REPO_ROOT / "bm25_indexes"
OUTPUT_PATH = REPO_ROOT / "retrieval_eval_results.json"

QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K = 10

VARIANTS = {
    "plain": "recursive_plain",
    "contextual": "recursive_contextual",
}


def load_valid_questions():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    valid = [q for q in questions if q["matched_chunk_indices"]]
    skipped = len(questions) - len(valid)
    print(f"Loaded {len(questions)} questions, {len(valid)} have verified ground truth "
          f"({skipped} skipped - no verified chunk to score against)")
    return valid


def rank_of_correct_hit(results, lecture_number, correct_chunk_indices):
    """results: list of (lecture_number, chunk_index) in rank order. Returns 1-indexed rank or None."""
    for rank, (lec, idx) in enumerate(results, start=1):
        if lec == lecture_number and idx in correct_chunk_indices:
            return rank
    return None


def vector_search(client, model, collection_name, query_text):
    query_vector = model.encode([query_text], normalize_embeddings=True)[0]
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=TOP_K,
        with_payload=True,
    )
    return [(h.payload["lecture_number"], h.payload["chunk_index"]) for h in response.points]


def bm25_search(retriever, query_text):
    nodes = retriever.retrieve(query_text)
    return [(n.metadata["lecture_number"], n.metadata["chunk_index"]) for n in nodes[:TOP_K]]


def score_retriever(name, ranks):
    n = len(ranks)
    hit_at_1 = sum(1 for r in ranks if r is not None and r <= 1) / n
    hit_at_5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    hit_at_10 = sum(1 for r in ranks if r is not None and r <= 10) / n
    mrr = sum(1 / r if r is not None else 0 for r in ranks) / n
    return {
        "retriever": name,
        "num_questions": n,
        "hit_rate@1": round(hit_at_1, 4),
        "hit_rate@5": round(hit_at_5, 4),
        "hit_rate@10": round(hit_at_10, 4),
        "mrr": round(mrr, 4),
    }


def main():
    questions = load_valid_questions()

    print(f"\nLoading {EMBEDDING_MODEL} locally for query embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    qdrant = QdrantClient(url=QDRANT_URL)

    print("Loading BM25 indexes...")
    bm25_retrievers = {
        variant: BM25Retriever.from_persist_dir(str(BM25_DIR / collection_name))
        for variant, collection_name in VARIANTS.items()
    }
    for retriever in bm25_retrievers.values():
        retriever.similarity_top_k = TOP_K

    ranks_by_retriever = {
        "vector_plain": [],
        "vector_contextual": [],
        "bm25_plain": [],
        "bm25_contextual": [],
    }
    per_question_detail = []

    for q in questions:
        correct = set(q["matched_chunk_indices"])
        lecture_number = q["lecture_number"]
        print(f"  scoring question {q['question_id']}: {q['question'][:70]}...")

        detail = {"question_id": q["question_id"], "question": q["question"]}
        for variant, collection_name in VARIANTS.items():
            results = vector_search(qdrant, model, collection_name, q["question"])
            rank = rank_of_correct_hit(results, lecture_number, correct)
            ranks_by_retriever[f"vector_{variant}"].append(rank)
            detail[f"vector_{variant}_rank"] = rank

        for variant, retriever in bm25_retrievers.items():
            results = bm25_search(retriever, q["question"])
            rank = rank_of_correct_hit(results, lecture_number, correct)
            ranks_by_retriever[f"bm25_{variant}"].append(rank)
            detail[f"bm25_{variant}_rank"] = rank

        per_question_detail.append(detail)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    summary = [score_retriever(name, ranks) for name, ranks in ranks_by_retriever.items()]
    for s in summary:
        print(s)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_question_detail}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
