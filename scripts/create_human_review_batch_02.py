import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "datasets/ranking/human_review_batch_01.csv"
CATALOG = ROOT / "datasets/ranking/richer_synthetic_catalog.csv"
OUTPUT = ROOT / "datasets/ranking/human_review_batch_02.csv"

with QUERIES.open("r", encoding="utf-8-sig", newline="") as f:
    query_rows = list(csv.DictReader(f))

queries = {}
for row in query_rows:
    queries.setdefault(row["query_id"], row["preference_text"])

with CATALOG.open("r", encoding="utf-8-sig", newline="") as f:
    catalog_rows = list(csv.DictReader(f))

# Deduplicate catalog entries by service name, preserving the first entry.
unique_catalog = {}
for service in catalog_rows:
    unique_catalog.setdefault(service["service_name"], service)

fields = [
    "query_id", "preference_text", "service_name", "category",
    "description", "label", "label_source", "review_status",
    "catalog_provenance",
]

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for query_id, preference in queries.items():
        for service in unique_catalog.values():
            writer.writerow({
                "query_id": query_id,
                "preference_text": preference,
                "service_name": service["service_name"],
                "category": service["category"],
                "description": service["description"],
                "label": "",
                "label_source": "",
                "review_status": "pending",
                "catalog_provenance": service["catalog_provenance"],
            })

print(f"Created: {OUTPUT.relative_to(ROOT)}")
print(f"Queries: {len(queries)}")
print(f"Unique services: {len(unique_catalog)}")
print(f"Candidate rows: {len(queries) * len(unique_catalog)}")
print("All labels remain blank.")
print("Source catalog and earlier review batches were not modified.")
