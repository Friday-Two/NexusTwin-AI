import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval.toolbench_ranker import get_retriever

LABELS_PATH = ROOT / "scripts" / "retrieval_labeling_sheet.csv"
OUT_PATH = ROOT / "scripts" / "geographic_reranking_full_evaluation.csv"


def geo_adjustment(item):
    text = (
        str(item.get("description", "")) + " " +
        str(item.get("api_text", ""))
    ).lower()

    strong = (
        "search businesses near",
        "search nearby places",
        "search for nearby places",
        "returns nearby places sorted by distance",
        "find places nearby",
        "nearby search",
    )
    helper = (
        "categories of establishments",
        "place details",
        "list of establishment types",
        "get a list of place types",
        "geocoding",
        "coordinates by placename",
        "coordinates to address",
        "random place",
        "random (google)",
        "air quality",
    )

    if any(term in text for term in helper):
        return 0.0
    if any(term in text for term in strong):
        return 0.30
    if any(term in text for term in (
        "nearby places", "search places", "places in radius",
    )):
        return 0.10
    return 0.0


def metrics(ranked_ids, labels):
    total = sum(labels.values())
    if total == 0:
        return np.nan, np.nan, np.nan

    # Preserve the actual ranking positions.
    ranked_labels = [
        labels.get(str(doc_id), 0)
        for doc_id in ranked_ids
    ]

    recall5 = sum(ranked_labels[:5]) / total

    first10 = next(
        (
            1.0 / rank
            for rank, value in enumerate(ranked_labels[:10], start=1)
            if value == 1
        ),
        0.0,
    )

    top10 = np.asarray(ranked_labels[:10], dtype=float)
    discounts = np.log2(np.arange(2, len(top10) + 2))
    dcg = float(np.sum(top10 / discounts))

    ideal_count = min(total, 10)
    idcg = float(
        np.sum(1.0 / np.log2(np.arange(2, ideal_count + 2)))
    )

    ndcg10 = dcg / idcg if idcg else 0.0
    return recall5, first10, ndcg10


def main():
    df = pd.read_csv(
        LABELS_PATH, dtype={"document_id": str}, encoding="cp1252"
    )
    df["relevant"] = pd.to_numeric(df["relevant"], errors="raise").astype(int)

    retriever = get_retriever()
    per_query = []

    for query, group in df.groupby("query", sort=False):
        labels = dict(zip(group["document_id"].astype(str), group["relevant"]))
        if not sum(labels.values()):
            print(f"Skipping zero-positive query: {query}")
            continue

        # Hybrid baseline
        baseline = retriever.rank(query, top_k=100)
        baseline_ids = [str(x["document_id"]) for x in baseline]

        # Geographic adjustment is applied only to queries that appear geographic.
        q = query.lower()
        geographic = any(word in q for word in (
            "nearby", "geographic", "coordinates", "location", "places"
        ))

        adjusted = []
        for item in baseline:
            row = dict(item)
            row["document_id"] = str(item["document_id"])
            row["adjusted_score"] = float(item["ranking_score"])
            if geographic:
                row["adjusted_score"] += geo_adjustment(item)
            adjusted.append(row)

        adjusted.sort(key=lambda x: x["adjusted_score"], reverse=True)
        adjusted_ids = [str(x["document_id"]) for x in adjusted]

        for method, ids in (
            ("Hybrid", baseline_ids),
            ("Geographic reranking", adjusted_ids),
        ):
            r5, mrr10, ndcg10 = metrics(ids, labels)
            per_query.append({
                "query": query,
                "method": method,
                "relevant_label_count": sum(labels.values()),
                "Recall@5": r5,
                "MRR@10": mrr10,
                "NDCG@10": ndcg10,
                "Relevant found@20": sum(
                    labels.get(doc_id, 0) for doc_id in ids[:20]
                ),
            })

    results = pd.DataFrame(per_query)
    results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== Mean metrics across positive-label queries ===")
    print(
        results.groupby("method")[
            ["Recall@5", "MRR@10", "NDCG@10", "Relevant found@20"]
        ].mean().round(4).to_string()
    )

    print("\n=== Per-query comparison ===")
    print(
        results.pivot(
            index="query",
            columns="method",
            values="NDCG@10",
        ).round(4).to_string()
    )
    print(f"\nSaved: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()