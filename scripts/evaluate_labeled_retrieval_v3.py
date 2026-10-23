import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval.toolbench_ranker import get_retriever

LABELS_PATH = ROOT / "scripts" / "retrieval_labeling_sheet.csv"


def dcg(rels, k):
    rels = np.asarray(rels[:k], dtype=float)
    if not len(rels):
        return 0.0
    return float(np.sum(rels / np.log2(np.arange(2, len(rels) + 2))))


def evaluate(ranked_ids, labels, k=10):
    # Keep original ranks. Unjudged documents are unknown, not negatives.
    judged = [labels.get(doc_id) for doc_id in ranked_ids[:k]]
    judged_count = sum(value is not None for value in judged)
    relevant_total = sum(value == 1 for value in labels.values())

    relevant_ranks = [
        i for i, value in enumerate(judged, start=1) if value == 1
    ]
    mrr = 1 / relevant_ranks[0] if relevant_ranks else 0.0

    # NDCG over judged results only, preserving their original rank discounts.
    gains = [value if value is not None else 0 for value in judged]
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = dcg(ideal, k)
    ndcg = dcg(gains, k) / idcg if idcg else np.nan

    recall = (
        sum(value == 1 for value in judged[:5]) / relevant_total
        if relevant_total else np.nan
    )

    return recall, mrr, ndcg, judged_count


def main():
    try:
        df = pd.read_csv(LABELS_PATH, dtype={"document_id": str},
                         encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(LABELS_PATH, dtype={"document_id": str},
                         encoding="cp1252")

    df["relevant"] = pd.to_numeric(df["relevant"], errors="raise").astype(int)
    if df.duplicated(["query", "document_id"]).any():
        raise ValueError("Duplicate query/document labels found.")
    if not df["relevant"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    retriever = get_retriever()
    rows = []

    for query, group in df.groupby("query", sort=False):
        labels = dict(zip(group["document_id"], group["relevant"]))
        total_relevant = sum(labels.values())

        for method_name, method in (
            ("Neural-only", retriever.rank_neural_only),
            ("Hybrid", retriever.rank),
        ):
            results = method(query, top_k=20)
            ranked_ids = [str(item["document_id"]) for item in results]
            recall, mrr, ndcg, judged = evaluate(ranked_ids, labels)

            rows.append({
                "query": query,
                "method": method_name,
                "labelled_relevant": total_relevant,
                "Recall@5": recall,
                "MRR@10": mrr,
                "NDCG@10_partial_judgments": ndcg,
                "Judged_in_top10": judged,
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(ROOT / "scripts" / "retrieval_evaluation_v3_per_query.csv",
                  index=False)

    summary = detail.groupby("method").agg(
        Recall_at_5=("Recall@5", "mean"),
        MRR_at_10=("MRR@10", "mean"),
        NDCG_at_10_partial_judgments=("NDCG@10_partial_judgments", "mean"),
        Mean_judged_in_top10=("Judged_in_top10", "mean"),
    )
    summary.to_csv(ROOT / "scripts" / "retrieval_evaluation_summary_v3.csv")
    print(summary.to_string())
    print("\nSaved v3 summary and per-query results under scripts/.")


if __name__ == "__main__":
    main()
