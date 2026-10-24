import csv

from scripts.export_reviewed_dataset import export_reviewed


def test_exports_only_valid_reviewed_rows(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"

    fields = [
        "query_id", "preference_text", "service_name",
        "category", "description", "label",
        "label_source", "review_status",
    ]
    rows = [
        {
            "query_id": "q1",
            "preference_text": "cheap hotel",
            "service_name": "Budget Hotel",
            "category": "Accommodation",
            "description": "Low cost rooms",
            "label": "1",
            "label_source": "human_review",
            "review_status": "reviewed",
        },
        {
            "query_id": "q1",
            "preference_text": "cheap hotel",
            "service_name": "Luxury Hotel",
            "category": "Accommodation",
            "description": "Premium rooms",
            "label": "",
            "label_source": "",
            "review_status": "pending",
        },
        {
            "query_id": "q2",
            "preference_text": "cheap taxi",
            "service_name": "Taxi",
            "category": "Transport",
            "description": "Low cost ride",
            "label": "0",
            "label_source": "human_review",
            "review_status": "reviewed",
        },
    ]

    with source.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    export_reviewed(source, output)

    with output.open("r", newline="", encoding="utf-8") as f:
        exported = list(csv.DictReader(f))

    assert len(exported) == 2
    assert {row["label"] for row in exported} == {"0", "1"}
    assert all(row["label_source"] == "human_review" for row in exported)
    assert "review_status" not in exported[0]


def test_export_fails_when_no_valid_rows(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"

    fields = [
        "query_id", "preference_text", "service_name",
        "category", "description", "label",
        "label_source", "review_status",
    ]

    with source.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "query_id": "q1",
            "preference_text": "cheap hotel",
            "service_name": "Hotel",
            "category": "Accommodation",
            "description": "Budget rooms",
            "label": "",
            "label_source": "",
            "review_status": "pending",
        })

    import pytest

    with pytest.raises(ValueError, match="No valid reviewed rows"):
        export_reviewed(source, output)
