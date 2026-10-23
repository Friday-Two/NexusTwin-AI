import sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from core.retrieval.toolbench_ranker import get_retriever

QUERY = "Find nearby places using geographic coordinates"
r = get_retriever()

def geo_adjustment(item):
    text = (
        str(item.get("description", "")) + " " +
        str(item.get("api_text", ""))
    ).lower()

    # Explicit nearby-result search is the strongest signal.
    strong = (
        "search businesses near",
        "search nearby places",
        "search for nearby places",
        "returns nearby places sorted by distance",
        "find places nearby",
        "nearby search",
    )
    if any(term in text for term in strong):
        score = 0.30
    elif any(term in text for term in (
        "nearby places", "search places", "places in radius",
    )):
        score = 0.10
    else:
        score = 0.0

    # Helper endpoints should not receive a search bonus.
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
        score = min(score, 0.0)

    return score

labels_path = ROOT / "scripts" / "retrieval_labeling_sheet.csv"
labels = pd.read_csv(labels_path, dtype={"document_id": str}, encoding="cp1252")
labels["relevant"] = pd.to_numeric(labels["relevant"], errors="coerce").fillna(0).astype(int)
relevant_ids = set(
    labels.loc[
        (labels["query"] == QUERY) & (labels["relevant"] == 1),
        "document_id"
    ].astype(str)
)

# Use a broad candidate pool for the experiment; production is untouched.
candidates = r.rank(QUERY, top_k=100)
baseline = r.rank(QUERY, top_k=20)

rows = []
for item in candidates:
    row = dict(item)
    row["document_id"] = str(item["document_id"])
    row["geo_adjustment"] = geo_adjustment(item)
    row["adjusted_score"] = float(item["ranking_score"]) + row["geo_adjustment"]
    row["relevant"] = int(row["document_id"] in relevant_ids)
    rows.append(row)

adjusted = sorted(rows, key=lambda x: x["adjusted_score"], reverse=True)

def show(title, items):
    print(f"\n=== {title} ===")
    for i, item in enumerate(items[:20], 1):
        doc_id = str(item["document_id"])
        relevant = int(doc_id in relevant_ids)
        adjustment = item.get("geo_adjustment", 0.0)
        desc = str(item.get("description", ""))[:100]
        print(f"{i:2}. ID={doc_id} rel={relevant} bonus={adjustment:+.2f} {desc}")

show("BASELINE HYBRID TOP 20", baseline)
show("REFINED EXPERIMENT TOP 20", adjusted)

for k in (5, 10, 20):
    base_ids = [str(x["document_id"]) for x in baseline[:k]]
    new_ids = [str(x["document_id"]) for x in adjusted[:k]]
    base_hits = sum(x in relevant_ids for x in base_ids)
    new_hits = sum(x in relevant_ids for x in new_ids)
    total = len(relevant_ids)
    print(
        f"Recall@{k}: baseline={base_hits}/{total} ({base_hits/total if total else 0:.3f}) "
        f"refined={new_hits}/{total} ({new_hits/total if total else 0:.3f})"
    )

out = pd.DataFrame(adjusted)
out.to_csv(
    ROOT / "scripts" / "geographic_reranking_refined.csv",
    index=False, encoding="utf-8-sig"
)
print("\nSaved: scripts/geographic_reranking_refined.csv")
