# Preference Ranking Datasets

## Dataset inventory

| File | Purpose | Research status |
|---|---|---|
| `starter_examples.csv` | Small synthetic training-pipeline smoke test | Not a benchmark |
| `expanded_synthetic_examples.csv` | Expanded synthetic candidate-ranking examples | Not a benchmark |
| `human_review_template.csv` | Blank-label review template | Pending review |
| `human_review_batch_01.csv` | First candidate-review batch | Pending review |
| `human_review_test.csv` | Script-generated labels used to test the workflow | Test data only |
| `human_reviewed_examples.csv` | Exported sample labels from the workflow test | Not independent human evidence |

## Expected columns

Core ranking fields:

- `query_id`: grouping key; keep all candidates for one query in the same split.
- `preference_text`: natural-language request or preference.
- `service_name`, `category`, `description`: candidate service fields.
- `label`: `1` for relevant, `0` for irrelevant, blank when not yet judged.
- `label_source`: `human_review` only after a genuine human judgment is recorded.
- `review_status`: `pending` or `reviewed`.

## Labeling procedure

1. Read the preference text and candidate description.
2. Assign `1` when the candidate satisfies the stated request.
3. Assign `0` when the candidate does not satisfy the request.
4. Leave the label blank and status `pending` if the available information is insufficient.
5. For a genuine human judgment, set `label_source` to `human_review` and `review_status` to `reviewed`.
6. Run `scripts/validate_human_review_dataset.py` and export eligible rows with `scripts/export_reviewed_dataset.py`.

Do not mark scripted labels as human judgments. The files `human_review_test.csv` and `human_reviewed_examples.csv` contain workflow-test labels and must not be treated as independent human-reviewed benchmark data.

## Research requirements

Before reporting model-quality results:

- Use a cited, legally usable benchmark or genuinely collected human judgments.
- Record the dataset source, license, collection procedure, preprocessing, and split strategy.
- Keep all candidates for the same query in one split to prevent query leakage.
- Use held-out queries for final evaluation.
- Compare the deterministic and ML rankers on the same candidates and labels.
- Report appropriate ranking metrics, including Recall@1, Recall@5, MRR@10, and NDCG@5.
- Do not enable ML ranking by default based only on synthetic labels.
- Avoid collecting unnecessary sensitive personal information.

## Current limitation

The existing synthetic datasets and scripted review examples validate the software workflow. They do not establish real-world relevance or production ranking quality. No independent human-reviewed benchmark is established by these files alone.
