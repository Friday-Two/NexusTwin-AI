
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/ranking/human_review_batch_02.csv"
OUTPUT = ROOT / "datasets/ranking/human_review_batch_04.csv"

RULES = {
    "q01": ({"Accommodation"}, {"hotel", "hostel", "inn", "guest", "airport"}),
    "q02": ({"Transport"}, {"taxi", "transfer", "airport", "ride"}),
    "q03": ({"Shopping"}, {"laptop", "student", "budget"}),
    "q04": ({"Food"}, {"vegetarian", "food", "meal", "lunch", "delivery"}),
    "q05": ({"Travel"}, {"train", "travel", "ticket", "planning"}),
    "q06": ({"Accommodation"}, {"premium", "luxury", "hotel", "comfortable"}),
    "q07": ({"Accommodation"}, {"budget", "student", "hotel", "affordable"}),
    "q08": ({"Transport"}, {"taxi", "ride", "airport", "transfer"}),
    "q09": ({"Food"}, {"restaurant", "meal", "delivery", "local"}),
    "q10": ({"Shopping"}, {"laptop", "accessories", "store"}),
    "q11": ({"Travel"}, {"train", "trip", "travel", "ticket", "cheap"}),
    "q12": ({"Accommodation"}, {"quiet", "hotel", "airport", "inn"}),
    "q13": ({"Transport"}, {"transport", "taxi", "city", "budget"}),
    "q14": ({"Food"}, {"healthy", "vegetarian", "lunch", "meal"}),
    "q15": ({"Accommodation"}, {"stay", "station", "guest", "house"}),
    "q16": ({"Travel"}, {"booking", "bus", "train", "travel", "ticket"}),
}

def tokens(text):
    return set(re.findall(r"[a-z]+", (text or "").lower()))

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
    if query_id not in RULES:
        raise ValueError(f"No selection rule defined for {query_id}")

    categories, keywords = RULES[query_id]
    preference = candidates[0]["preference_text"]

    def score(row):
        text = tokens(
            row["service_name"] + " " + row["description"]
        )
        return (
            4 * int(row["category"] in categories)
            + 2 * len(text & keywords)
            + len(tokens(preference) & text)
        )

    ranked = sorted(
        candidates,
        key=lambda row: (score(row), row["service_name"].lower()),
        reverse=True,
    )

    matching = [r for r in ranked if score(r) > 0]
    unrelated = [r for r in ranked if score(r) == 0]

    chosen = matching[:4]
    if unrelated:
        chosen.append(unrelated[-1])

    for row in ranked:
        if len(chosen) >= 5:
            break
        if row not in chosen:
            chosen.append(row)

    if len(chosen) != min(5, len(candidates)):
        raise ValueError(f"Unexpected candidate count for {query_id}")

    for row in chosen:
        row["label"] = ""
        row["label_source"] = ""
        row["review_status"] = "pending"
        selected.append(row)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(selected)

print(f"Created: {OUTPUT.relative_to(ROOT)}")
print(f"Queries: {len(groups)}")
print(f"Rows: {len(selected)}")
print("All labels remain blank and pending.")
print("Selection is rule-based; human review is still required.")
