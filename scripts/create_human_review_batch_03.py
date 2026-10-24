
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/ranking/human_review_batch_02.csv"
OUTPUT = ROOT / "datasets/ranking/human_review_batch_03.csv"

STOP_WORDS = {
    "a", "an", "the", "in", "near", "under", "for", "with",
    "and", "or", "to", "of", "on", "my", "me", "want",
    "need", "find", "looking", "cheap", "best", "please",
}

def tokens(text):
    return set(re.findall(r"[a-z]+", (text or "").lower())) - STOP_WORDS

def score(preference, row):
    query = tokens(preference)
    name = tokens(row["service_name"])
    category = tokens(row["category"])
    description = tokens(row["description"])

    return (
        3 * len(query & name)
        + 2 * len(query & category)
        + len(query & description)
    )

with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("Source dataset is empty.")

groups = {}
for row in rows:
    groups.setdefault(row["query_id"], []).append(row)

fields = list(rows[0].keys())
selected = []

for query_id, candidates in sorted(groups.items()):
    preference = candidates[0]["preference_text"]

    ranked = sorted(
        candidates,
        key=lambda row: (
            score(preference, row),
            row["service_name"].lower(),
        ),
        reverse=True,
    )

    stronger = [r for r in ranked if score(preference, r) > 0]
    weaker = [r for r in ranked if score(preference, r) == 0]

    chosen = stronger[:4]
    if weaker:
        chosen.append(weaker[-1])

    for row in ranked:
        if len(chosen) >= 5:
            break
        if row not in chosen:
            chosen.append(row)

    for row in chosen[:5]:
        row["label"] = ""
        row["label_source"] = ""
        row["review_status"] = "pending"

        if "catalog_provenance" in fields:
            row["catalog_provenance"] = (
                row.get("catalog_provenance")
                or "synthetic; attributes are illustrative and unverified"
            )

        selected.append(row)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(selected)

print(f"Created: {OUTPUT.relative_to(ROOT)}")
print(f"Queries: {len(groups)}")
print(f"Rows: {len(selected)}")
print(f"Candidates per query: {len(selected) // len(groups)}")
print("Labels remain blank; all rows are pending.")
print("Scores prioritize lexical overlap, not verified relevance.")
