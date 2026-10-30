# Configuration

Runs are driven by two YAML files, merged into one dict at startup (`run.py`).

| File | Role |
|---|---|
| `config_model.yaml` | FAF-only model, optimiser and training: the INR decoder (architecture, `latent_dim`, `hidden_size`, SIREN `omega_*` schedule, `out_dim = [#intensity, #seg]`, segmentation head), the optimiser (`lr_inr`, `lr_latent`, loss weights, test-time optimisation early stopping), `epochs`, `validate_every`, `n_samples` and `model_gen`. Its top-level `config_data:` key names the dataset section to use. The shipped values are the paper's. `config_model_faf_slo_ga.yaml` is the joint FAF+SLO+GA variant; `optimizer.tto_reconstruction_modalities` selects observed-visit TTO channels independently of `seg_loss_val`. |
| `config_data.yaml` | Dataset sections: CSV, ordered modalities and explicit `modality_roles`, conditioning variables and their constraints, the temporal variable, mask grader mode, and image geometry (`crop_before_resize`, `world_bbox`, `sampling_bbox`). `faf_ga_twovar_wktemporal_512` is the FAF-only training section and `faf_slo_ga_twovar_wktemporal_512` the multimodal section; the FAF `_256`, `_620` and `_512_to256` sections support evaluation at other grids (`evaluate.py --config_data`). |
| `subject_ids.yaml` | Optional explicit train/val/test ID lists. The `faf_ga` lists are empty, so the loader uses the CSV `split` column instead. |
| `expected_split.yaml` | The paper split as explicit eye-ID lists (26 / 5 / 6). `validate_splits()` aborts if a run resolves a different partition. |
| `lakefs_cfg.example.yaml` | Template for the optional lakeFS credentials. Copy it to `lakefs_cfg.yaml` (git-ignored) and fill it in, or ignore it to read local files. |

Edit `tsv_file` in `config_data.yaml` to point at your CSV (see [`../docs/DATA.md`](../docs/DATA.md)).
Write floats in scientific notation with a decimal point (`1.0e-4`, not `1e-4`): PyYAML reads `1e-4`
as a string.
