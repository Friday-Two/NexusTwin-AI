import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "datasets" / "ranking" / "human_review_template.csv"


def validate(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "query_id", "preference_text", "service_name",
            "category", "description", "label",
            "label_source", "review_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Dataset is empty.")

    errors = []
    seen = set()
    query_texts = defaultdict(set)
    query_labels = defaultdict(list)

    for line, row in enumerate(rows, start=2):
        query_id = row["query_id"].strip()
        service = row["service_name"].strip()
        key = (query_id, service)

        if not all(row.get(c, "").strip() for c in (
            "query_id", "preference_text", "service_name",
            "category", "description",
        )):
            errors.append(f"Line {line}: missing required text.")

        if key in seen:
            errors.append(f"Line {line}: duplicate query/service pair {key}.")
        seen.add(key)

        query_texts[query_id].add(row["preference_text"].strip())
        label = row["label"].strip()
        status = row["review_status"].strip().lower()
        source = row["label_source"].strip()

        if status == "pending":
            if label or source:
                errors.append(
                    f"Line {line}: pending rows must have blank label/source."
                )
        elif status == "reviewed":
            if label not in {"0", "1"}:
                errors.append(f"Line {line}: reviewed label must be 0 or 1.")
            if source != "human_review":
                errors.append(
                    f"Line {line}: reviewed row needs label_source=human_review."
                )
        else:
            errors.append(
                f"Line {line}: review_status must be pending or reviewed."
            )

        if status == "reviewed" and label in {"0", "1"}:
            query_labels[query_id].append(label)

    for query_id, texts in query_texts.items():
        if len(texts) > 1:
            errors.append(f"{query_id}: inconsistent preference text.")

    if errors:
        print(f"Validation failed: {len(errors)} issue(s)")
        for error in errors[:30]:
            print("-", error)
        if len(errors) > 30:
            print(f"... and {len(errors) - 30} more")
        return False

    reviewed = sum(
        row["review_status"].strip().lower() == "reviewed"
        for row in rows
    )
    pending = len(rows) - reviewed
    complete_queries = sum(
        "0" in labels and "1" in labels
        for labels in query_labels.values()
    )

    print("Validation passed.")
    print(f"Total rows: {len(rows)}")
    print(f"Reviewed: {reviewed}")
    print(f"Pending: {pending}")
    print(f"Queries with reviewed positive and negative labels: {complete_queries}")
    print("Note: structural validation does not establish label quality.")
    return True


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    if not target.is_absolute():
        target = ROOT / target
    try:
        success = validate(target)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    sys.exit(0 if success else 1)
