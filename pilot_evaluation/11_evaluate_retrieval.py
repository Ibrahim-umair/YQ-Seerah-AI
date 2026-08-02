"""
The full retrieval evaluation for the 10-lecture pilot: embeds
eval_questions.json's questions and scores all 6 retriever conditions in
one pass:

    - vector_plain / vector_contextual        (BGE-M3, local)
    - bm25_plain / bm25_contextual             (keyword)
    - vector_openai_large_plain / _contextual  (OpenAI text-embedding-3-large)

(Originally two scripts - 11 for BGE-M3+BM25, 15 for OpenAI large - merged
into one since they score the same question set against the same pilot
data, just different collections.)

A "hit" means the correct chunk (per matched_chunk_indices in
eval_questions.json) appeared in a retriever's top-k results, for the same
lecture_number. Questions with no verified ground truth (empty
matched_chunk_indices) are skipped, not counted as misses.

Metrics: Hit Rate @1 / @5 / @10, and MRR (Mean Reciprocal Rank).

Output: retrieval_eval_results.json
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from qdrant_client import QdrantClient
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv()
openai_client = OpenAI()

LOCAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = LOCAL_DIR / "eval_questions.json"
BM25_DIR = LOCAL_DIR / "bm25_indexes"
OUTPUT_PATH = LOCAL_DIR / "retrieval_eval_results.json"

QDRANT_URL = "http://localhost:6333"
BGE_MODEL = "BAAI/bge-m3"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 10

BGE_VARIANTS = {"plain": "recursive_plain", "contextual": "recursive_contextual"}
OPENAI_VARIANTS = {"plain": "recursive_plain_openai_large", "contextual": "recursive_contextual_openai_large"}


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


def bge_vector_search(qdrant, model, collection_name, query_text):
    query_vector = model.encode([query_text], normalize_embeddings=True)[0]
    response = qdrant.query_points(collection_name=collection_name, query=query_vector.tolist(), limit=TOP_K, with_payload=True)
    return [(h.payload["lecture_number"], h.payload["chunk_index"]) for h in response.points]


def openai_vector_search(qdrant, collection_name, query_vector):
    response = qdrant.query_points(collection_name=collection_name, query=query_vector, limit=TOP_K, with_payload=True)
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
        "retriever": name, "num_questions": n,
        "hit_rate@1": round(hit_at_1, 4), "hit_rate@5": round(hit_at_5, 4),
        "hit_rate@10": round(hit_at_10, 4), "mrr": round(mrr, 4),
    }


def main():
    questions = load_valid_questions()

    print(f"\nLoading {BGE_MODEL} locally for query embeddings...")
    bge_model = SentenceTransformer(BGE_MODEL)
    qdrant = QdrantClient(url=QDRANT_URL, timeout=30)

    print("Loading BM25 indexes...")
    bm25_retrievers = {variant: BM25Retriever.from_persist_dir(str(BM25_DIR / name)) for variant, name in BGE_VARIANTS.items()}
    for retriever in bm25_retrievers.values():
        retriever.similarity_top_k = TOP_K

    print("Pre-embedding questions with OpenAI large (reused across both OpenAI collections)...")
    openai_query_vectors = {
        q["question_id"]: openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=[q["question"]]).data[0].embedding
        for q in questions
    }

    ranks_by_retriever = {
        "vector_plain": [], "vector_contextual": [],
        "bm25_plain": [], "bm25_contextual": [],
        "vector_openai_large_plain": [], "vector_openai_large_contextual": [],
    }
    per_question_detail = []

    for q in questions:
        correct = set(q["matched_chunk_indices"])
        lecture_number = q["lecture_number"]
        print(f"  scoring question {q['question_id']}: {q['question'][:70]}...")

        detail = {"question_id": q["question_id"], "question": q["question"]}

        for variant, collection_name in BGE_VARIANTS.items():
            results = bge_vector_search(qdrant, bge_model, collection_name, q["question"])
            rank = rank_of_correct_hit(results, lecture_number, correct)
            ranks_by_retriever[f"vector_{variant}"].append(rank)
            detail[f"vector_{variant}_rank"] = rank

        for variant, retriever in bm25_retrievers.items():
            results = bm25_search(retriever, q["question"])
            rank = rank_of_correct_hit(results, lecture_number, correct)
            ranks_by_retriever[f"bm25_{variant}"].append(rank)
            detail[f"bm25_{variant}_rank"] = rank

        for variant, collection_name in OPENAI_VARIANTS.items():
            results = openai_vector_search(qdrant, collection_name, openai_query_vectors[q["question_id"]])
            rank = rank_of_correct_hit(results, lecture_number, correct)
            ranks_by_retriever[f"vector_openai_large_{variant}"].append(rank)
            detail[f"vector_openai_large_{variant}_rank"] = rank

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
