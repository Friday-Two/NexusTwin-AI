
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/ranking/human_review_batch_02.csv"
CATALOG = ROOT / "datasets/ranking/richer_synthetic_catalog.csv"
OUTPUT = ROOT / "datasets/ranking/human_review_batch_05.csv"

EXPECTED = {
    "q01": "Accommodation",
    "q02": "Transport",
    "q03": "Shopping",
    "q04": "Food",
    "q05": "Travel",
    "q06": "Accommodation",
    "q07": "Accommodation",
    "q08": "Transport",
    "q09": "Food",
    "q10": "Shopping",
    "q11": "Travel",
    "q12": "Accommodation",
    "q13": "Transport",
    "q14": "Food",
    "q15": "Accommodation",
    "q16": "Travel",
}

STOP = {
    "a", "an", "the", "in", "near", "under", "for", "with",
    "and", "or", "to", "of", "on", "my", "me", "want",
    "need", "find", "looking", "help",
}

def tokens(text):
    return set(re.findall(r"[a-z]+", (text or "").lower())) - STOP

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    source_rows = list(csv.DictReader(f))

with CATALOG.open(encoding="utf-8-sig", newline="") as f:
    catalog = list(csv.DictReader(f))

if not source_rows or not catalog:
    raise ValueError("Source queries and catalog must not be empty.")

queries = {}
for row in source_rows:
    queries.setdefault(row["query_id"], row["preference_text"])

catalog_names = [r["service_name"].strip().casefold() for r in catalog]
if len(catalog_names) != len(set(catalog_names)):
    raise ValueError("Catalog contains duplicate service names.")

fields = [
    "query_id", "preference_text", "service_name", "category",
    "description", "label", "label_source", "review_status",
    "catalog_provenance",
]
selected = []

for query_id, preference in sorted(queries.items()):
    if query_id not in EXPECTED:
        raise ValueError(f"No expected category for {query_id}")

    category = EXPECTED[query_id]
    candidates = [r for r in catalog if r["category"] == category]

    if len(candidates) < 5:
        raise ValueError(f"Fewer than five catalog candidates for {category}")

    query_tokens = tokens(preference)

    ranked = sorted(
        candidates,
        key=lambda r: (
            len(query_tokens & tokens(
                r["service_name"] + " " + r["description"]
            )),
            r["service_name"].casefold(),
        ),
        reverse=True,
    )

    for row in ranked[:5]:
        selected.append({
            "query_id": query_id,
            "preference_text": preference,
            "service_name": row["service_name"],
            "category": row["category"],
            "description": row["description"],
            "label": "",
            "label_source": "",
            "review_status": "pending",
            "catalog_provenance": row["catalog_provenance"],
        })

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(selected)

print(f"Created: {OUTPUT.relative_to(ROOT)}")
print(f"Queries: {len(queries)}")
print(f"Rows: {len(selected)}")
print("All candidates match their query's expected category.")
print("Labels remain blank and pending.")
print("Selection uses lexical similarity, not verified relevance.")
