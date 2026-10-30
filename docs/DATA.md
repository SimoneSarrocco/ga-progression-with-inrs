# Data preparation

How to put your own longitudinal FAF / geographic-atrophy (GA) cohort into GAP-INR. It covers the
metadata CSV and how to construct the conditioning variables, the folder structure for the image and
mask files (local and lakeFS), the preprocessing chain, and the train/val/test split. The dataset
used in the paper is private and is not distributed; any cohort that matches the format below will
run.

One sample is one visit of one patient-eye: a fundus-autofluorescence (FAF) image and a GA
segmentation mask. The patient-eye is the identity unit and receives one latent code, shared across
all of its visits; visits are told apart by a temporal conditioning variable (weeks from baseline).

---

## 1. The metadata CSV (`tsv_file`)

A single comma-separated table, **one row per patient-eye visit**, pointed to by
`tsv_file` in `configs/config_data.yaml` (default `./data/clinical_metadata.csv`). The loader reads:

| Column | Required | Type | Meaning / how to construct |
|---|---|---|---|
| `Eye_ID` | **yes** | str | Patient-eye identity (`id_column`); the key mapping to one latent code. Identical across that eye's visits, such as `EYE07_OD`. Convention: suffix `_OD` (right) or `_OS` (left), used for laterality (section 4). |
| `split` | **yes** | str | `train` / `val` / `test` (`split_column`). Defines the partition (section 6). Build with `preprocessing/add_split_column.py` (patient-wise). |
| `Patient_ID` | **yes** | str | Patient identifier; used to build image/mask paths and group eyes. |
| `Eye` | **yes** | str | `OD` / `OS`; used for path reconstruction and laterality. |
| `Visit_ID` | **yes** | str/int | Visit id, e.g. `V01`, or an integer `1` (formatted to `V01`). Used to build the mask paths. |
| `Visit_Number` | **yes** | int | Ordinal visit index (1, 2, 3, …). Names the held-out visit in the metric files (`EYE07_OD_V3`), which `summarise_eval.py` parses to group visits by eye; also used by the fixed-schedule time option and to fill undated visits (section 3). |
| `faf_path` | **yes** | str | Location of the FAF image (section 5). |
| `slo_path` | multimodal only | str | Location of the registered SLO image. Required by `faf_slo_ga_*` sections; local and lakeFS resolution use `Spectralis_slo`. |
| `ga_mask_path` | recommended | str | Path of the grader-1 mask (`…_mask01.png`). The loader finds masks through the fixed layout of section 5, not through this column, but with `mask_grader_mode: single` the split guard drops rows where it is empty, and the growth buckets of `summarise_eval.py` and the baselines read it. |
| `visit_date` / `Visit_date` | recommended | number or date | Per-visit date; numeric = Excel serial day, else a parseable date. Source for `weeks_from_baseline` (section 3-A). |
| `diff` | alt. to date | number | Elapsed **days** from the eye's baseline visit (section 3-B). |
| `AgeatVisit` | **yes** (shipped config) | float | Age (years) at the visit. A FiLM condition in the shipped config, so rows without it are dropped. It can also serve as the time axis (`temporal_condition: AgeatVisit`). |
| `ScaleXSlo`, `ScaleYSlo` | recommended | float | Physical pixel size (mm/pixel), to convert GA pixel counts to mm². Default `1.0` if absent. |
| `lesion_size`, … | optional | float | Any extra numeric column can be enabled as a conditioning/constraint variable (section 2). |

> Eye IDs in `configs/expected_split.yaml` are anonymized placeholders (`EYE01_OD …`). Replace that
> file (and the CSV) with your own IDs, or regenerate the split (section 6).

`modalities` defines the ordered target columns; `modality_roles` defines how each column is used:

```yaml
modalities: ['faf_path', 'slo_path', 'ga_mask_path']
modality_roles:
  faf_path: reconstruction
  slo_path: reconstruction
  ga_mask_path: segmentation
primary_reconstruction_modality: faf_path
```

Every listed modality is required: a visit is dropped if any file is missing. Multimodal images must be
registered to the same field of view before loading; the loader applies shared crop/resize/flip and
coordinate sampling but does not perform cross-modal registration. Old checkpoints without
`modality_roles` retain the legacy reconstruction-plus-final-mask interpretation.

---

## 2. Conditions vs. constraints (config-driven)

The INR is conditioned on a **temporal variable** and, optionally, extra covariates. Declared in the
dataset section of `configs/config_data.yaml`:

```yaml
faf_ga:
  conditions:
    AgeatVisit: true           # FiLM condition
    weeks_from_baseline: true  # FiLM condition
  temporal_condition: weeks_from_baseline   # which variable is the time axis
  constraints:                              # [min, max] per variable
    weeks_from_baseline: {type: numeric, min: 0.0, max: 54.143}
    AgeatVisit:          {type: numeric, min: 66.0, max: 91.0}
```

This is the shipped configuration: both variables are FiLM conditions.

- `conditions: <name>: true` feeds the variable to the decoder as a FiLM condition.
- `temporal_condition` names the time axis: the variable that orders an eye's visits (which one is
  held out as "last") and that is moved to render other time points.
- `constraints` give the `[min, max]` used to normalise each variable to `[-1, 1]`. Rows whose value
  is outside the range, or missing, are dropped when the data is loaded. When a visit at a new time
  point is rendered, values past the range are clamped to it unless `extrapolate_beyond_range: true`
  (the shipped setting).

Every extra covariate must be a CSV column and be listed under both `conditions` and `constraints`.

---

## 3. Constructing the temporal variable `weeks_from_baseline`

Computed automatically at load time (`_add_weeks_from_baseline_col`) from the first available of:

- **A, dates (recommended):** `visit_date`/`Visit_date`. Numeric values are Excel serial **days**;
  anything else is parsed as a date. `weeks = (date - eye's earliest date) / 7`.
- **B, elapsed days:** a `diff` column (days from baseline). `weeks = diff / 7`.
- **C, fixed schedule:** if neither of the above is present, `visit_week_map` in `config_data.yaml`
  maps `Visit_Number` to weeks, e.g. `{1: 0, 2: 12, 3: 24, 4: 48}`.

With A or B, visits without a date get their week from `visit_week_map` (or 0 if the map has no
entry), so they are not dropped. With `weeks_constraint_from_dates: true` (shipped), the normalisation
`[min, max]` is the range of `weeks_from_baseline` over the whole CSV, all splits included; the value
in the YAML is then ignored. `weeks_from_baseline` is never read from the CSV.

---

## 4. Laterality

`canonicalize_laterality: true` in the dataset section mirrors left (OS) eyes to the right (OD)
orientation, with the same horizontal flip for image and mask. Pixel areas are unchanged. Left eyes are
detected from the `Eye_ID` suffix (`…_OS`) or the `Eye`/`Laterality` column. It is off in the shipped
configuration (the key is absent).

---

## 5. Folder structure for images and masks

**lakeFS is optional** (see section 7). The two modes below differ only in *where the `data/` tree lives*.

### 5.1 The canonical `data/` tree

For every visit, the GA grader masks are found through this fixed layout, built from `Patient_ID`,
`Eye` and `Visit_ID`:

```
data/<Patient_ID>/<Eye>/<Vxx>/Spectralis_faf/<Patient_ID>_<Eye>_<Vxx>_mask01.png
data/<Patient_ID>/<Eye>/<Vxx>/Spectralis_faf/<Patient_ID>_<Eye>_<Vxx>_mask02.png   # optional (majority/soft)
data/<Patient_ID>/<Eye>/<Vxx>/Spectralis_faf/<Patient_ID>_<Eye>_<Vxx>_mask03.png   # optional (majority/soft)
```

`mask_grader_mode` (in `config_data.yaml`) selects how they combine:

| Mode | Behaviour |
|---|---|
| `single` | use `…_mask01.png` only (**use this if you have one mask per visit**) |
| `majority` (paper default) | per-pixel vote over the available `mask01/02/03`: GA where at least half of them mark GA (so with two graders, either one); written next to them as `…_majority.png` and reused on later runs |
| `soft` | per-pixel mean of graders (soft training target; evaluation uses the majority vote) |

Masks are grayscale PNGs. The majority vote counts a grader pixel as GA above 127; the mask the model
reads is then binarised at > 0.

The **FAF image** is taken from the `faf_path` CSV column (any 2-D grayscale `.png`/`.bmp`). Keep it
in the same tree for tidiness:

```
data/<Patient_ID>/<Eye>/<Vxx>/Spectralis_faf/<Patient_ID>_<Eye>_<Vxx>_FAF.png
```

### 5.2 LOCAL setup (no lakeFS), the recommended default

Masks are resolved at `<cache_path>/<branch>/data/…`. With the defaults
`cache_path: ./cache/faf_ga` and `branch: main` (from the `lakefs:` block of `config_data.yaml`), the
tree is:

```
<repo>/
├── cache/faf_ga/main/data/<Patient_ID>/<Eye>/<Vxx>/Spectralis_faf/
│        ├── <Patient_ID>_<Eye>_<Vxx>_mask01.png      # (+ mask02/03 for majority/soft)
│        └── <Patient_ID>_<Eye>_<Vxx>_FAF.png         # FAF here too (optional but tidy)
└── data/clinical_metadata.csv                        # the CSV (tsv_file)
```

Concretely, set in the CSV: `faf_path = ./cache/faf_ga/main/data/<Patient_ID>/<Eye>/<Vxx>/Spectralis_faf/<Patient_ID>_<Eye>_<Vxx>_FAF.png`
(an existing absolute path is used as is; a relative path is resolved from the repository root,
where the commands run). The FAF image may live anywhere as long as `faf_path` points to it; only the
masks are bound to the `cache_path/branch/data/…` layout above. To keep everything under another
root, change `cache_path` (e.g. `cache_path: /data/my_cohort` puts the masks under
`/data/my_cohort/main/data/…`).

### 5.3 lakeFS setup (remote store)

The lakeFS repository holds the same `data/<Patient_ID>/<Eye>/<Vxx>/Spectralis_faf/…` object layout on
the chosen branch. On first use, objects are downloaded to `<cache_path>/<branch>/data/…` (same local
tree as section 5.2). Configure:

```yaml
# configs/config_data.yaml  (faf_ga block)
lakefs:
  repo: your-lakefs-repo         # lakeFS repository name
  branch: main                   # branch holding the registered images
  cache_path: ./cache/faf_ga     # local download cache
```

Then supply the credentials. Copy the template and fill in four values:

```bash
cp configs/lakefs_cfg.example.yaml configs/lakefs_cfg.yaml
```

```yaml
# configs/lakefs_cfg.yaml   (git-ignored, never commit it)
endpoint:    "https://my-lakefs-host:8000"   # the S3 gateway, not the web UI
access_key:  "..."                           # lakeFS UI: Administration -> Access Keys
secret_key:  "..."                           # shown once, when the key is created
ca_bundle:   ""                              # CA file for TLS; empty disables verification
```

Every one of these can come from the environment instead, which is usually better on a cluster or in
CI. Environment values win over the file:

| Variable | Replaces |
|---|---|
| `GAPINR_LAKEFS_ENDPOINT` | `endpoint` |
| `GAPINR_LAKEFS_ACCESS_KEY` | `access_key` |
| `GAPINR_LAKEFS_SECRET_KEY` | `secret_key` |
| `GAPINR_LAKEFS_CA_BUNDLE` | `ca_bundle` |
| `GAPINR_LAKEFS_REPO` / `_BRANCH` / `_CACHE` | the `lakefs:` block above |
| `GAPINR_LAKEFS_CFG` | the path of `lakefs_cfg.yaml` |

Check the settings before starting a long run:

```bash
python download_lakefs_data.py --check      # prints what it resolved, makes one request
python download_lakefs_data.py              # pre-fetch every image into the cache
```

`--check` never prints your secret key. All three of `endpoint`, `access_key` and `secret_key` are
required: set none of them to read from local disk, and setting only some is reported as an error
straight away rather than as an obscure S3 failure later.

lakeFS object keys mirror the local tree: `data/<Patient_ID>/<Eye>/<Vxx>/Spectralis_faf/<file>`, and
downloads land at `<cache_path>/<branch>/data/…`, where the local-disk path in section 5.2
looks. So a cache filled over lakeFS keeps working when you later run with no credentials.

---

## 6. Train / validation / test split

The split is the `split` column of the CSV. `configs/subject_ids.yaml` keeps the `faf_ga` lists
empty, so the loader reads the CSV column, which the baselines read as well. Eyes with fewer than
`min_valid_visits_eval` (2) usable visits are dropped from val/test (leave-one-out needs at least
two).

`preprocessing/add_split_column.py` writes a patient-wise split (both eyes of a patient in the same
split; default 70/15/15, seed 42). The paper's partition is frozen in `configs/expected_split.yaml`
(26 train / 5 val / 6 test eyes), and `validate_splits()` aborts every run if the resolved partition
differs from it or if train/val/test overlap. Replace `expected_split.yaml` for your own cohort.

---

## 7. Do I need lakeFS? No.

lakeFS is an **optional** remote-store convenience, off by default:

- With no credentials the loader prints `[lakeFS] not configured, reading images from local disk`
  and reads every file from local disk (section 5.2). This is the default. Only the `.example`
  template ships.
- To use a remote store, follow section 5.3.
- You can also delete the `lakefs:` block; masks then resolve under `./cache/faf_ga/main/data/…`.

Credentials are never committed (`lakefs_cfg.yaml` is git-ignored).

---

## 8. Preprocessing chain (raw image to model input)

Performed in `data_loading/dataset.py`:

1. **Load** configured reconstruction images and GA mask as 2-D grayscale (`PIL`).
2. **Centre-crop, then resize:** crop the native image (768 px) to `crop_before_resize` (620 px),
   which removes the black registration frame and keeps all GA, then resize to `world_bbox` (512 px):
   FAF/SLO bilinear, mask nearest. The resize changes the pixel size; the area conversion below corrects
   for it.
3. **Binarise** the mask to `{0, 1}`.
4. **Coordinate sampling** (`sampling_strategy: all`): every pixel becomes a coordinate in `[-1, 1]`.
5. **Intensity normalisation** (`normalize_values: minmax`, paper default): each visit's FAF and,
   when configured, SLO are independently min-max scaled to `[0, 1]`.
6. **Conditioning:** `weeks_from_baseline` and `AgeatVisit` are attached per visit, normalised to
   `[-1, 1]` with the `constraints` ranges.

Lesion area (metrics/figures) is `#GA pixels × ScaleXSlo × ScaleYSlo × rf` mm², identical for
prediction and the identically-cropped ground truth. `ScaleXSlo`/`ScaleYSlo` are mm per pixel at
*native* resolution, so `rf = (crop_before_resize / world_bbox)²` converts them to the scoring grid:
resizing the 620 crop to 512 makes each grid pixel cover `(620/512)² ≈ 1.47` native pixels of area.
`rf = 1` for sections that crop at native pitch instead of resizing. `build_model.py`'s
`_lesion_px_area_mm2` computes it for every area GAP-INR reports, and the ImageFlowNet baselines
apply the same factor for their grid.

---

## 9. Building the CSV from scratch (optional helpers)

The `preprocessing/` scripts document how the paper's CSV was assembled from a raw export. They are
reference material to adapt, not a chain to run in order, and they overwrite
`data/clinical_metadata.csv`, including its `split` column. Read
[`../preprocessing/README.md`](../preprocessing/README.md) before running any of them.

- `enrich_clinical_data.py`: merge per-scan physical-scale metadata into the raw CSV
- `add_split_column.py`: draw the patient-wise train/val/test `split` column

---

## 10. Checklist for your own cohort

1. Build `data/clinical_metadata.csv` with the columns in section 1 (one row per eye-visit).
2. Provide `visit_date`, `diff`, or `Visit_Number` + `visit_week_map` for the time axis (section 3).
3. Lay out FAF images and grader masks per section 5 (5.2 local, 5.3 lakeFS) and pick
   `mask_grader_mode` (`single` for one mask per visit).
4. Add the `split` column and regenerate `configs/expected_split.yaml` for your eye IDs (section 6).
5. Point `tsv_file` at your CSV; adjust `crop_before_resize`/`world_bbox` to your image size if needed.
6. Run `python run.py` (see the main [README](../README.md)).
