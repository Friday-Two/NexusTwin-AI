import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "datasets/ranking/human_review_batch_01.csv"
output = root / "datasets/ranking/human_review_test.csv"

labels = {
    ("q01", "Budget Airport Hotel"): 1,
    ("q01", "Premium Gaming Laptop"): 0,
    ("q02", "Airport Taxi"): 1,
    ("q02", "Luxury City Hotel"): 0,
    ("q03", "Student Laptop Store"): 1,
    ("q03", "Premium City Hotel"): 0,
}

with source.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

selected = []
for row in rows:
    key = (row["query_id"], row["service_name"])
    if key in labels:
        row["label"] = str(labels[key])
        row["label_source"] = "human_review"
        row["review_status"] = "reviewed"
        selected.append(row)

if len(selected) != len(labels):
    raise ValueError(f"Expected {len(labels)} rows, found {len(selected)}")

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(selected)

print(f"Created: {output.relative_to(root)}")
print(f"Reviewed rows: {len(selected)}")
print(f"Positive labels: {sum(int(r['label']) for r in selected)}")
print(f"Negative labels: {sum(int(r['label']) == 0 for r in selected)}")
