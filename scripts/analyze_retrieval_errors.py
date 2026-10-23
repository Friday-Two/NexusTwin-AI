
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval.toolbench_ranker import get_retriever

CSV_PATH = PROJECT_ROOT / "scripts" / "retrieval_labeling_sheet.csv"


def main():
    df = pd.read_csv(CSV_PATH, encoding="cp1252", dtype={"document_id": str})
    df["relevant"] = pd.to_numeric(df["relevant"], errors="raise").astype(int)

    retriever = get_retriever()
    rows = []

    for query, group in df.groupby("query", sort=False):
        labels = dict(zip(group["document_id"], group["relevant"]))
        relevant_ids = {
            api_id for api_id, label in labels.items() if label == 1
        }

        results = retriever.rank(query, top_k=10)
        ranked_ids = [str(result["document_id"]) for result in results]

        found = set(ranked_ids[:5]) & relevant_ids
        missed = relevant_ids - set(ranked_ids[:5])

        rows.append({
            "query": query,
            "labeled_relevant": len(relevant_ids),
            "relevant_in_top5": len(found),
            "recall_at_5": (
                len(found) / len(relevant_ids) if relevant_ids else 0
            ),
            "relevant_ids_missed_from_top5": ", ".join(sorted(missed)),
        })

    report = pd.DataFrame(rows).sort_values("recall_at_5")
    print(report.to_string(index=False))

    output = PROJECT_ROOT / "scripts" / "retrieval_error_analysis.csv"
    report.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
