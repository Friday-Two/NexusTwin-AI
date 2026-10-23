
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval.toolbench_ranker import get_retriever

CSV_PATH = PROJECT_ROOT / "scripts" / "retrieval_labeling_sheet.csv"


def recall_at_k(labels, total_relevant, k=5):
    if total_relevant == 0:
        return np.nan
    return sum(labels[:k]) / total_relevant


def mrr_at_k(labels, k=10):
    for rank, label in enumerate(labels[:k], start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels, k=10):
    labels = np.asarray(labels[:k], dtype=float)
    if len(labels) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(labels) + 2))
    dcg = float(np.sum(labels / discounts))
    ideal = np.sort(labels)[::-1]
    idcg = float(np.sum(ideal / discounts))

    return dcg / idcg if idcg else 0.0


def main():
    df = pd.read_csv(
        CSV_PATH,
        dtype={"document_id": str},
        encoding="cp1252",
    )
    df["relevant"] = pd.to_numeric(
        df["relevant"], errors="raise"
    ).astype(int)

    retriever = get_retriever()
    methods = {
        "Hybrid": retriever.rank,
        "Neural-first": retriever.rank_neural_first,
    }

    rows = []

    for query, group in df.groupby("query", sort=False):
        labels_by_id = dict(zip(
            group["document_id"].astype(str),
            group["relevant"],
        ))
        candidate_ids = set(labels_by_id)
        total_relevant = int(group["relevant"].sum())

        for method_name, method in methods.items():
            results = method(query, top_k=20)

            # Keep only the manually labelled candidates.
            ranked_ids = []
            seen = set()

            for result in results:
                api_id = str(result["document_id"])
                if api_id in candidate_ids and api_id not in seen:
                    ranked_ids.append(api_id)
                    seen.add(api_id)

            ranked_labels = [
                labels_by_id[api_id] for api_id in ranked_ids
            ]

            rows.append({
                "query": query,
                "method": method_name,
                "labelled_candidates_returned": len(ranked_ids),
                "relevant_total": total_relevant,
                "relevant_in_top5": sum(ranked_labels[:5]),
                "recall_at_5": recall_at_k(
                    ranked_labels, total_relevant, 5
                ),
                "mrr_at_10": mrr_at_k(ranked_labels, 10),
                "ndcg_at_10": ndcg_at_k(ranked_labels, 10),
            })

    details = pd.DataFrame(rows)

    print("\n=== Per-query comparison ===")
    print(details.round(4).to_string(index=False))

    valid = details.dropna(subset=["recall_at_5"])
    summary = valid.groupby("method").agg(
        Recall_at_5=("recall_at_5", "mean"),
        MRR_at_10=("mrr_at_10", "mean"),
        NDCG_at_10=("ndcg_at_10", "mean"),
        Mean_labelled_candidates=(
            "labelled_candidates_returned", "mean"
        ),
    )

    print("\n=== Method summary ===")
    print(summary.round(4).to_string())

    details.to_csv(
        PROJECT_ROOT / "scripts" / "ranking_methods_per_query.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        PROJECT_ROOT / "scripts" / "ranking_methods_summary.csv",
        encoding="utf-8-sig",
    )

    print("\nReports saved:")
    print("scripts/ranking_methods_per_query.csv")
    print("scripts/ranking_methods_summary.csv")


if __name__ == "__main__":
    main()
