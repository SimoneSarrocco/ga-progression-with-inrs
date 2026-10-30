# Data loading

| File | Contents |
|---|---|
| `dataset.py` | `Data(Dataset)`: resolves the split, loads role-labelled reconstruction images (FAF, optionally SLO) and GA masks (grader consensus), applies shared geometry and per-channel normalisation, samples coordinates, and attaches conditioning variables. `__getitem__` returns coordinate/value/condition tensors for one visit. Also `validate_splits` (the split guard), `EyeBatchSampler` (one batch per eye), `get_longitudinal_indices` (which visits are fitted and which are held out in a test-time optimisation round) and `sample_subject_ids` (split selection). |
| `lakefsloader.py` | `LakeFSLoader`: optional backend that fetches images from an S3-compatible object store (lakeFS) into a local cache on first use. |
| `lakefs_config.py` | Resolves the optional lakeFS connection from `configs/lakefs_cfg.yaml` or `GAPINR_LAKEFS_*` environment variables, for both `dataset.py` and `download_lakefs_data.py`. Returns `None` when lakeFS is not configured, so images are read from local disk, and raises an error naming the missing setting when it is only partly configured. `python download_lakefs_data.py --check` verifies the credentials without starting a run. |

Images are loaded as 2-D grayscale (PNG/BMP); the pipeline is 2-D only.

The exact data format, on-disk layout, preprocessing steps, and split mechanism are documented in
[`../docs/DATA.md`](../docs/DATA.md).
