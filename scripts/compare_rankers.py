
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval.toolbench_ranker import get_retriever


def recall_at_k(ranked_ids, relevant_ids, k=5):
    return (
        len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)
        if relevant_ids else 0.0
    )


def mrr_at_k(ranked_ids, relevant_ids, k=10):
    for rank, api_id in enumerate(ranked_ids[:k], start=1):
        if api_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def main():
    cases_path = PROJECT_ROOT / "scripts" / "retrieval_test_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    retriever = get_retriever()

    metrics = {
        "Neural-only": {"recall": [], "mrr": []},
        "Hybrid": {"recall": [], "mrr": []},
    }

    for case in cases:
        query = case["query"]
        relevant = {str(x) for x in case["relevant_api_ids"]}

        for name, method in (
            ("Neural-only", retriever.rank_neural_only),
            ("Hybrid", retriever.rank),
        ):
            results = method(query, top_k=10)
            ids = [str(item["document_id"]) for item in results]

            recall = recall_at_k(ids, relevant, 5)
            mrr = mrr_at_k(ids, relevant, 10)

            metrics[name]["recall"].append(recall)
            metrics[name]["mrr"].append(mrr)

            print(f"\n{name} | {query}")
            print(f"Recall@5={recall:.3f}, MRR@10={mrr:.3f}")
            print("Top 3 IDs:", ids[:3])

    print("\n=== MEAN COMPARISON ===")
    for name, values in metrics.items():
        print(
            f"{name:12s} "
            f"Recall@5={sum(values['recall']) / len(cases):.3f} "
            f"MRR@10={sum(values['mrr']) / len(cases):.3f}"
        )


if __name__ == "__main__":
    main()
