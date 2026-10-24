import csv

from scripts.validate_human_review_dataset import validate


FIELDS = [
    "query_id", "preference_text", "service_name",
    "category", "description", "label",
    "label_source", "review_status",
]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(query, preference, service, label="", source="", status="pending"):
    return {
        "query_id": query,
        "preference_text": preference,
        "service_name": service,
        "category": "Accommodation",
        "description": "Hotel rooms",
        "label": label,
        "label_source": source,
        "review_status": status,
    }


def test_valid_pending_rows_pass(tmp_path, capsys):
    path = tmp_path / "pending.csv"
    write_csv(path, [
        row("q1", "cheap hotel", "Budget Hotel"),
        row("q1", "cheap hotel", "Luxury Hotel"),
    ])

    assert validate(path) is True
    assert "Pending: 2" in capsys.readouterr().out


def test_inconsistent_query_text_fails(tmp_path, capsys):
    path = tmp_path / "inconsistent.csv"
    write_csv(path, [
        row("q1", "cheap hotel", "Budget Hotel"),
        row("q1", "luxury hotel", "Luxury Hotel"),
    ])

    assert validate(path) is False
    assert "inconsistent preference text" in capsys.readouterr().out


def test_duplicate_query_service_fails(tmp_path, capsys):
    path = tmp_path / "duplicates.csv"
    write_csv(path, [
        row("q1", "cheap hotel", "Budget Hotel"),
        row("q1", "cheap hotel", "Budget Hotel"),
    ])

    assert validate(path) is False
    assert "duplicate query/service pair" in capsys.readouterr().out
