from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval.toolbench_ranker import get_retriever

LABELS = ROOT / "scripts" / "retrieval_labeling_sheet.csv"
OUTPUT_DETAILS = ROOT / "scripts" / "candidate_coverage_per_query.csv"
OUTPUT_SUMMARY = ROOT / "scripts" / "candidate_coverage_summary.csv"


def main():
    df = pd.read_csv(
        LABELS,
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
        "Neural-only": retriever.rank_neural_only,
    }

    rows = []

    for query, group in df.groupby("query", sort=False):
        labels = dict(zip(
            group["document_id"].astype(str),
            group["relevant"],
        ))
        relevant_ids = {
            doc_id for doc_id, label in labels.items() if label == 1
        }

        for method_name, method in methods.items():
            for k in (5, 10, 20):
                results = method(query, top_k=k)
                returned_ids = {
                    str(item["document_id"]) for item in results
                }
                found_relevant = relevant_ids & returned_ids

                rows.append({
                    "query": query,
                    "method": method_name,
                    "k": k,
                    "labelled_relevant_total": len(relevant_ids),
                    "relevant_found": len(found_relevant),
                    "labelled_candidate_coverage": (
                        len(found_relevant) / len(relevant_ids)
                        if relevant_ids else float("nan")
                    ),
                    "returned_count": len(results),
                    "unique_returned_ids": len(returned_ids),
                })

    details = pd.DataFrame(rows)
    summary = (
        details.groupby(["method", "k"], as_index=False)
        .agg(
            queries=("query", "nunique"),
            queries_with_labels=(
                "labelled_relevant_total",
                lambda values: int((values > 0).sum()),
            ),
            mean_labelled_candidate_coverage=(
                "labelled_candidate_coverage", "mean"
            ),
            mean_relevant_found=("relevant_found", "mean"),
            mean_returned=("returned_count", "mean"),
            mean_unique_ids=("unique_returned_ids", "mean"),
        )
    )

    details.to_csv(OUTPUT_DETAILS, index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n=== Candidate coverage summary ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nDetailed results: {OUTPUT_DETAILS}")
    print(f"Summary results:  {OUTPUT_SUMMARY}")
    print(
        "\nNote: coverage is measured only against labelled relevant IDs. "
        "It does not measure relevance among the unlabelled corpus."
    )


if __name__ == "__main__":
    main()
