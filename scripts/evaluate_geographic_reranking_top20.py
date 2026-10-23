import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval.geographic_reranker import (
    is_geographic_query,
    rerank,
)
from core.retrieval.toolbench_ranker import get_retriever

LABELS_PATH = ROOT / "scripts" / "retrieval_labeling_sheet.csv"
OUT_PATH = ROOT / "scripts" / "geographic_reranking_top20_evaluation.csv"


def metrics(ranked_ids, labels):
    total = sum(labels.values())
    if total == 0:
        return np.nan, np.nan, np.nan, 0

    values = [labels.get(str(doc_id), 0) for doc_id in ranked_ids]
    recall5 = sum(values[:5]) / total

    mrr10 = next(
        (1.0 / rank for rank, value in enumerate(values[:10], 1) if value),
        0.0,
    )

    top10 = np.asarray(values[:10], dtype=float)
    discounts = np.log2(np.arange(2, len(top10) + 2))
    dcg = float(np.sum(top10 / discounts))

    ideal_count = min(total, 10)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_count + 2))))
    ndcg10 = dcg / idcg if idcg else 0.0

    return recall5, mrr10, ndcg10, sum(values[:20])


def main():
    df = pd.read_csv(
        LABELS_PATH,
        dtype={"document_id": str},
        encoding="cp1252",
    )
    df["relevant"] = pd.to_numeric(df["relevant"], errors="raise").astype(int)

    retriever = get_retriever()
    rows = []

    for query, group in df.groupby("query", sort=False):
        labels = dict(
            zip(
                group["document_id"].astype(str),
                group["relevant"],
            )
        )

        if not sum(labels.values()):
            print(f"Skipping zero-positive query: {query}")
            continue

        # Match the production default: top_k=5 requests up to 20 candidates.
        candidates = retriever.rank(query, top_k=20)
        baseline_ranked = [
            str(item["document_id"]) for item in candidates
        ]

        adjusted = rerank(query, candidates)
        adjusted_ranked = [
            str(item["document_id"]) for item in adjusted
        ]

        for method, ids in (
            ("Hybrid baseline", baseline_ranked),
            ("Geographic reranking", adjusted_ranked),
        ):
            recall5, mrr10, ndcg10, _ = metrics(ids, labels)

            found20 = sum(labels.get(doc_id, 0) for doc_id in ids[:20])

            rows.append({
                "query": query,
                "method": method,
                "positive_labels": sum(labels.values()),
                "Recall@5": recall5,
                "MRR@10": mrr10,
                "NDCG@10": ndcg10,
                "Relevant found@20": found20,
            })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== Mean metrics: positive-label queries ===")
    print(
        results.groupby("method")[
            ["Recall@5", "MRR@10", "NDCG@10", "Relevant found@20"]
        ]
        .mean()
        .round(4)
        .to_string()
    )

    print("\n=== Per-query NDCG@10 ===")
    print(
        results.pivot(
            index="query", columns="method", values="NDCG@10"
        )
        .round(4)
        .to_string()
    )

    print(f"\nSaved: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()