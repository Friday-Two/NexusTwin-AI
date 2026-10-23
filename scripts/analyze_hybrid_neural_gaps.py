from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retrieval.toolbench_ranker import get_retriever

labels_path = ROOT / "scripts" / "retrieval_labeling_sheet.csv"
output_path = ROOT / "scripts" / "hybrid_neural_missed_candidates.csv"

df = pd.read_csv(
    labels_path,
    dtype={"document_id": str},
    encoding="cp1252",
)
df["relevant"] = pd.to_numeric(df["relevant"], errors="raise").astype(int)

retriever = get_retriever()
rows = []

for query, group in df.groupby("query", sort=False):
    relevant = group[group["relevant"] == 1].drop_duplicates("document_id")
    if relevant.empty:
        continue

    hybrid = retriever.rank(query, top_k=20)
    neural = retriever.rank_neural_first(query, top_k=20)

    hybrid_ids = {str(x["document_id"]) for x in hybrid}
    neural_ids = {str(x["document_id"]) for x in neural}

    missed_ids = (hybrid_ids - neural_ids) & set(relevant["document_id"])

    for doc_id in sorted(missed_ids):
        match = relevant[relevant["document_id"] == doc_id].iloc[0]
        rows.append({
            "query": query,
            "document_id": doc_id,
            "api_text": match["api_text"],
            "hybrid_rank": next(
                i for i, x in enumerate(hybrid, 1)
                if str(x["document_id"]) == doc_id
            ),
            "neural_first_rank": "Not in top 20",
        })

report = pd.DataFrame(rows)
report.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Relevant Hybrid candidates missed by Neural-first: {len(report)}")
if not report.empty:
    print(report[["query", "document_id", "hybrid_rank"]].to_string(index=False))
else:
    print("No such candidates found in this labelled set.")

print(f"\nReport saved to: {output_path}")
