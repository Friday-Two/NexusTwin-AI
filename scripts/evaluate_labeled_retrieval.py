
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval.toolbench_ranker import get_retriever


CSV_PATH = PROJECT_ROOT / "scripts" / "retrieval_labeling_sheet.csv"


def recall_at_k(relevant, k=5):
    return float(np.sum(relevant[:k]) / np.sum(relevant)) if np.sum(relevant) else 0.0


def mrr_at_k(relevant, k=10):
    for rank, value in enumerate(relevant[:k], start=1):
        if value:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant, k=10):
    relevant = np.asarray(relevant[:k], dtype=float)
    discounts = np.log2(np.arange(2, len(relevant) + 2))
    dcg = np.sum(relevant / discounts)

    ideal = np.sort(relevant)[::-1]
    idcg = np.sum(ideal / discounts)
    return float(dcg / idcg) if idcg else 0.0


def main():
    try:
      df = pd.read_csv(
        CSV_PATH,
        dtype={"document_id": str},
        encoding="utf-8-sig",
    )
    except UnicodeDecodeError:
      df = pd.read_csv(
        CSV_PATH,
        dtype={"document_id": str},
        encoding="cp1252",
    )
    df["relevant"] = pd.to_numeric(df["relevant"], errors="raise").astype(int)

    if not df["relevant"].isin([0, 1]).all():
        raise ValueError("The relevant column must contain only 0 or 1.")

    retriever = get_retriever()
    summaries = {}

    # Each query's labels are aligned to the candidate IDs in the CSV.
    for method_name, method in [
        ("Neural-only", retriever.rank_neural_only),
        ("Hybrid", retriever.rank),
    ]:
        recalls, mrrs, ndcgs = [], [], []

        for query, group in df.groupby("query", sort=False):
            labels = dict(zip(
                group["document_id"].astype(str),
                group["relevant"],
            ))

            results = method(query, top_k=10)
            ranked_labels = [
                labels.get(str(result["document_id"]), 0)
                for result in results
            ]

            # Only evaluate the labeled candidate pool.
            # Candidates outside this pool are not treated as verified negatives.
            ranked_labels = [
                labels[str(result["document_id"])]
                for result in results
                if str(result["document_id"]) in labels
            ]

            total_relevant = int(group["relevant"].sum())
            recalls.append(
                sum(ranked_labels[:5]) / total_relevant
                if total_relevant else 0.0
            )
            mrrs.append(mrr_at_k(ranked_labels))
            ndcgs.append(ndcg_at_k(ranked_labels))

        summaries[method_name] = {
            "Recall@5": float(np.mean(recalls)),
            "MRR@10": float(np.mean(mrrs)),
            "NDCG@10": float(np.mean(ndcgs)),
        }

    print("\n=== Labeled Retrieval Comparison ===")
    for method_name, metrics in summaries.items():
        print(f"\n{method_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    output = pd.DataFrame(summaries).T
    output.to_csv(
        PROJECT_ROOT / "scripts" / "retrieval_evaluation_summary.csv"
    )
    print("\nSaved: scripts/retrieval_evaluation_summary.csv")


if __name__ == "__main__":
    main()
