import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "datasets" / "ranking" / "expanded_synthetic_examples.csv"
output = root / "datasets" / "ranking" / "human_review_template.csv"

with source.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

fields = [
    "query_id",
    "preference_text",
    "service_name",
    "category",
    "description",
    "label",
    "label_source",
    "review_status",
]

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "query_id": row["query_id"],
            "preference_text": row["preference_text"],
            "service_name": row["service_name"],
            "category": row["category"],
            "description": row["description"],
            "label": "",
            "label_source": "",
            "review_status": "pending",
        })

print(f"Created: {output.relative_to(root)}")
print(f"Rows requiring review: {len(rows)}")
print("Labels are intentionally blank; no synthetic labels were copied.")
