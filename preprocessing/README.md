# Preprocessing

Scripts that assembled the clinical metadata table and the train/val/test split the training code
reads. They are cohort-specific and document how the paper's CSV was built; adapt the column names and
paths (`./data/...`) to your own data before running them.

| Script | What it does |
|---|---|
| `enrich_clinical_data.py` | Merges per-scan physical-scale metadata (SLO scan-info) into the longitudinal clinical CSV. |
| `add_split_column.py` | Adds a patient-wise `split` column (train/val/test) so no eye leaks across splits. |

These are **not** a pipeline to run against a CSV you already have, and they are destructive:

- `enrich_clinical_data.py` reads `./data/clinical_metadata_raw.csv` (a raw export, not shipped) and
  writes `./data/clinical_metadata.csv`, adding the physical pixel-scale columns (`ScaleXSlo`,
  `ScaleYSlo`, …) that convert GA pixel counts to mm².
- `add_split_column.py` rewrites `./data/clinical_metadata.csv` **in place** with a freshly drawn
  split. Running it on a CSV that already carries the paper's `split` column replaces that column,
  after which `validate_splits` aborts every run against `configs/expected_split.yaml`. It refuses to
  be imported (it would rewrite the CSV at import time), so run it as a script or not at all.

Do not run them against a CSV that already has the columns of [`../docs/DATA.md`](../docs/DATA.md):
that CSV is already the finished table that `tsv_file` in `configs/config_data.yaml` points at.
