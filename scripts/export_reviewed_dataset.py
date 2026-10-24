import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "query_id", "preference_text", "service_name",
    "category", "description", "label",
    "label_source", "review_status",
}


def export_reviewed(source, output):
    with source.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        rows = list(reader)

    reviewed = [
        row for row in rows
        if row["review_status"].strip().lower() == "reviewed"
        and row["label"].strip() in {"0", "1"}
        and row["label_source"].strip() == "human_review"
    ]

    if not reviewed:
        raise ValueError("No valid reviewed rows to export.")

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id", "preference_text", "service_name",
        "category", "description", "label", "label_source",
    ]

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in reviewed)

    try:
        display_path = output.relative_to(ROOT)
    except ValueError:
        display_path = output
    print(f"Exported: {display_path}")
    print(f"Reviewed rows exported: {len(reviewed)}")
    print(f"Positive: {sum(row['label'] == '1' for row in reviewed)}")
    print(f"Negative: {sum(row['label'] == '0' for row in reviewed)}")


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "datasets/ranking/human_review_template.csv"
    )
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        ROOT / "datasets/ranking/human_reviewed_examples.csv"
    )
    if not source.is_absolute():
        source = ROOT / source
    if not output.is_absolute():
        output = ROOT / output

    try:
        export_reviewed(source, output)
    except (OSError, ValueError) as exc:
        print(f"Export failed: {exc}")
        sys.exit(1)
