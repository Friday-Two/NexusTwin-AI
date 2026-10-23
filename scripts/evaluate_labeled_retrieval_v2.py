import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval.toolbench_ranker import get_retriever

CSV_PATH = PROJECT_ROOT / "scripts" / "retrieval_labeling_sheet.csv"


def recall_at_k(labels, k=5):
    total_relevant = int(np.sum(labels))
    if total_relevant == 0:
        return np.nan
    return float(np.sum(labels[:k]) / total_relevant)


def mrr_at_k(labels, k=10):
    for rank, value in enumerate(labels[:k], start=1):
        if value == 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels, k=10):
    labels = np.asarray(labels[:k], dtype=float)
    if not len(labels):
        return 0.0

    discounts = np.log2(np.arange(2, len(labels) + 2))
    dcg = float(np.sum(labels / discounts))
    ideal = np.sort(labels)[::-1]
    idcg = float(np.sum(ideal / discounts))

    return dcg / idcg if idcg else 0.0


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

    required = {"query", "document_id", "relevant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df["relevant"] = pd.to_numeric(
        df["relevant"], errors="raise"
    ).astype(int)

    if not df["relevant"].isin([0, 1]).all():
        raise ValueError("The relevant column must contain only 0 or 1.")

    retriever = get_retriever()
    summaries = {}

    for method_name, method in [
        ("Neural-only", retriever.rank_neural_only),
        ("Hybrid", retriever.rank),
    ]:
        recalls, mrrs, ndcgs = [], [], []
        queries_without_positives = 0

        for query, group in df.groupby("query", sort=False):
            # Restrict ranking to IDs that were actually labelled.
            labels = dict(zip(
                group["document_id"].astype(str),
                group["relevant"],
            ))
            labelled_ids = set(labels)

            results = method(query, top_k=20)
            ranked_labels = [
                labels[str(result["document_id"])]
                for result in results
                if str(result["document_id"]) in labelled_ids
            ]

            total_relevant = int(group["relevant"].sum())

            if total_relevant == 0:
                queries_without_positives += 1
                mrrs.append(mrr_at_k(ranked_labels))
                ndcgs.append(ndcg_at_k(ranked_labels))
                continue

            recalls.append(recall_at_k(ranked_labels, k=5))
            mrrs.append(mrr_at_k(ranked_labels, k=10))
            ndcgs.append(ndcg_at_k(ranked_labels, k=10))

        summaries[method_name] = {
            "Recall@5": float(np.nanmean(recalls)) if recalls else np.nan,
            "MRR@10": float(np.mean(mrrs)) if mrrs else np.nan,
            "NDCG@10": float(np.mean(ndcgs)) if ndcgs else np.nan,
            "Queries without positives": queries_without_positives,
        }

    print("\n=== Labelled-pool Retrieval Comparison ===")
    for method_name, metrics in summaries.items():
        print(f"\n{method_name}")
        for metric, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                print(f"{metric}: {value}")
            elif np.isfinite(value):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: N/A")

    output = pd.DataFrame(summaries).T
    output.to_csv(
        PROJECT_ROOT / "scripts" / "retrieval_evaluation_summary_v2.csv"
    )
    print("\nSaved: scripts/retrieval_evaluation_summary_v2.csv")


if __name__ == "__main__":
    main()