# Preference ranking dataset

`starter_examples.csv` is a **small synthetic smoke-test dataset** created to validate the training pipeline. It is not a public benchmark and must not be used to claim benchmark-level model quality.

## Expected columns

- `query_id`: grouping key; keep all candidates for one query in the same split.
- `preference_text`: natural-language request/preferences.
- `service_name`, `category`, `description`: candidate service fields.
- `label`: 1 for relevant, 0 for not relevant.

## Before reporting research results

Replace or augment these rows with a cited, legally usable real benchmark or consented user preference data. Record dataset name, source URL, license, preprocessing, and split strategy. Avoid user-sensitive attributes. Split by query/user group to prevent leakage. Report metrics on a held-out benchmark split, not the synthetic starter file.
