import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "ranking" / "human_review_template.csv"
OUTPUT = ROOT / "datasets" / "ranking" / "human_review_batch_01.csv"

with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

# Select up to 30 queries, taking the first two candidates per query.
selected = []
seen_queries = set()
counts = {}

for row in rows:
    qid = row["query_id"]
    if qid not in seen_queries:
        if len(seen_queries) >= 30:
            continue
        seen_queries.add(qid)
    if qid in seen_queries and counts.get(qid, 0) < 2:
        selected.append(row)
        counts[qid] = counts.get(qid, 0) + 1

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(selected)

print(f"Created: {OUTPUT.relative_to(ROOT)}")
print(f"Queries: {len(seen_queries)}")
print(f"Candidates: {len(selected)}")
print("All labels remain blank for independent review.")
