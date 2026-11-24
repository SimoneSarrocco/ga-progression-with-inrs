# Preprocessing

Scripts that assembled the clinical metadata table and the train/val/test split the training code
reads. They are cohort-specific and document how the paper's CSV was built; adapt the column names and
paths (`./data/...`) to your own data before running them.

| Script | What it does |
|---|---|
| `enrich_clinical_data.py` | Merges per-scan physical-scale metadata (SLO scan-info) into the longitudinal clinical CSV. |
| `add_split_column.py` | Adds a patient-wise `split` column (train/val/test) so no eye leaks across splits. |
| `prepare_macustar_faf.py` | Builds reconstruction-only MACUSTAR metadata from the validated survival table and master ages, preserving patient splits and resolving age gaps. Also writes `expected_split_macustar.yaml`. It reads registered FAF from `enhanced_registered_multimodal` and does not include survival labels. |
| `analyze_macustar_biomarker_survival.py` | Extracts the 15 total OCT biomarker volumes from the registered-branch JSON files and fits baseline study-eye Cox models for late-AMD progression. Reports presence and log-volume hazard ratios with age/sex adjustment, baseline-AMD stratification, and within-family FDR correction. |
| `extract_macustar_biomarker_volumes.py` | Extracts all 15 total volumes for every available study and fellow eye visit into `biomarker_volumes_all_visits.csv`, checks each volume against voxel count × voxel volume, and records missing/invalid JSON files. The companion RStudio workflow is `../analysis/macustar_biomarker_cox.R`. |

The two OMEGA scripts below are **not** a pipeline to run against a CSV you already have, and they are destructive:

- `enrich_clinical_data.py` reads `./data/clinical_metadata_raw.csv` (a raw export, not shipped) and
  writes `./data/clinical_metadata.csv`, adding the physical pixel-scale columns (`ScaleXSlo`,
  `ScaleYSlo`, …) that convert GA pixel counts to mm².
- `add_split_column.py` rewrites `./data/clinical_metadata.csv` **in place** with a freshly drawn
  split. Running it on a CSV that already carries the paper's `split` column replaces that column,
  after which `validate_splits` aborts every run against `configs/expected_split.yaml`. It refuses to
  be imported (it would rewrite the CSV at import time), so run it as a script or not at all.

Do not run them against a CSV that already has the columns of [`../docs/DATA.md`](../docs/DATA.md):
that CSV is already the finished table that `tsv_file` in `configs/config_data.yaml` points at.
