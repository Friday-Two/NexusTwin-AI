
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval.toolbench_ranker import get_retriever


TEST_CASES_PATH = PROJECT_ROOT / "scripts" / "retrieval_test_cases.json"


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(ranked_ids, relevant_ids, k):
    for rank, api_id in enumerate(ranked_ids[:k], start=1):
        if api_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    dcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank, api_id in enumerate(ranked_ids[:k], start=1)
        if api_id in relevant_ids
    )

    ideal_relevant = min(len(relevant_ids), k)
    idcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    return dcg / idcg if idcg else 0.0


def main():
    if not TEST_CASES_PATH.is_file():
        raise FileNotFoundError(
            f"Create the test case file first: {TEST_CASES_PATH}"
        )

    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    retriever = get_retriever()
    recall_values = []
    mrr_values = []
    ndcg_values = []

    for case in cases:
        query = case["query"]
        relevant_ids = {str(x) for x in case["relevant_api_ids"]}

        results = retriever.rank(query, top_k=10)
        ranked_ids = [str(result["document_id"]) for result in results]

        recall = recall_at_k(ranked_ids, relevant_ids, 5)
        mrr = reciprocal_rank_at_k(ranked_ids, relevant_ids, 10)
        ndcg = ndcg_at_k(ranked_ids, relevant_ids, 10)

        recall_values.append(recall)
        mrr_values.append(mrr)
        ndcg_values.append(ndcg)

        print(f"\nQuery: {query}")
        print(f"Recall@5: {recall:.3f}")
        print(f"MRR@10:   {mrr:.3f}")
        print(f"NDCG@10:  {ndcg:.3f}")

        for rank, result in enumerate(results[:5], start=1):
            print(
                f"  {rank}. ID={result['document_id']} | "
                f"{result['api_text'][:150]}"
            )

        missing = relevant_ids - set(ranked_ids)
        if missing:
            print("Relevant IDs not retrieved in top 10:", sorted(missing))

    count = len(cases)
    if count == 0:
        raise ValueError("The test case file contains no cases.")

    print("\n=== Mean Evaluation Metrics ===")
    print(f"Queries:  {count}")
    print(f"Recall@5: {sum(recall_values) / count:.3f}")
    print(f"MRR@10:   {sum(mrr_values) / count:.3f}")
    print(f"NDCG@10:  {sum(ndcg_values) / count:.3f}")
    print(
        "\nInterpretation depends on the quality and completeness "
        "of the manually verified relevant API IDs."
    )


if __name__ == "__main__":
    main()
