
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval.toolbench_ranker import get_retriever

CSV_PATH = PROJECT_ROOT / "scripts" / "retrieval_labeling_sheet.csv"


def recall_at_k(labels, k=5):
    total = int(np.sum(labels))
    return float(np.sum(labels[:k]) / total) if total else np.nan


def mrr_at_k(labels, k=10):
    for rank, label in enumerate(labels[:k], start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels, k=10):
    labels = np.asarray(labels[:k], dtype=float)
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
    rows = []

    for query, group in df.groupby("query", sort=False):
        labels_by_id = dict(zip(
            group["document_id"].astype(str),
            group["relevant"],
        ))
        candidate_ids = set(labels_by_id)

        for method_name, method in [
            ("Neural-only", retriever.rank_neural_only),
            ("Hybrid", retriever.rank),
        ]:
            # Retrieve the entire labelled pool, then retain only
            # manually labelled candidates.
            results = method(query, top_k=20)
            labelled_results = [
                item for item in results
                if str(item["document_id"]) in candidate_ids
            ]

            ranked_labels = [
                labels_by_id[str(item["document_id"])]
                for item in labelled_results
            ]

            relevant_total = int(group["relevant"].sum())
            rows.append({
                "query": query,
                "method": method_name,
                "labelled_candidates_returned": len(labelled_results),
                "relevant_total": relevant_total,
                "recall_at_5": recall_at_k(ranked_labels),
                "mrr_at_10": mrr_at_k(ranked_labels),
                "ndcg_at_10": ndcg_at_k(ranked_labels),
            })

    details = pd.DataFrame(rows)
    valid_recall = details.dropna(subset=["recall_at_5"])

    summary = valid_recall.groupby("method").agg(
        Recall_at_5=("recall_at_5", "mean"),
    ).join(
        details.groupby("method").agg(
            MRR_at_10=("mrr_at_10", "mean"),
            NDCG_at_10=("ndcg_at_10", "mean"),
            Mean_labelled_candidates_returned=(
                "labelled_candidates_returned", "mean"
            ),
        )
    )

    print("\n=== Labelled-candidate ranking comparison ===")
    print(summary.round(4).to_string())

    print("\n=== Per-query results ===")
    print(
        details.round(4).to_string(index=False)
    )

    details.to_csv(
        PROJECT_ROOT / "scripts" / "labeled_candidate_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        PROJECT_ROOT / "scripts" / "labeled_candidate_summary.csv",
        encoding="utf-8-sig",
    )

    print("\nSaved:")
    print("scripts/labeled_candidate_comparison.csv")
    print("scripts/labeled_candidate_summary.csv")


if __name__ == "__main__":
    main()
