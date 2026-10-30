# Reproducing the paper

This is the end-to-end recipe to reproduce the paper's numbers. It assumes the environment is set up
(`conda env create -f environment.yml && conda activate gap-inr`) and the data is in place per
[`DATA.md`](DATA.md).

## 0. Split and determinism

- The train/val/test partition (26 / 5 / 6 eyes) is fixed by the CSV `split` column and guarded by
  `configs/expected_split.yaml`; every run aborts on a mismatch.
- Seeds are set from the config (`seed: 1927` for the paper model), and every test-time optimisation
  round restarts from the same seed, so its latent initialisation is always the same.
- Training is not bit-for-bit reproducible on a GPU. The backward pass of `grid_sample` (the bilinear
  lookup of the latent grid) has no deterministic CUDA implementation, so two runs with the same
  seed and data drift apart slightly from the first epoch on (epoch-0 training loss 0.2074 vs 0.2080
  in two such runs), and so do repeated test-time optimisations.
- To make an evaluation repeatable, `evaluate.py` saves the optimised latents of every round to
  `evaluation_*/test_latents.pt`. Running it again with `--test_latents <that file>` loads them,
  skips the optimisation, and gives the same numbers. `--test_latents off` disables this.
- Choose models on the validation metric and evaluate the test set once, for the final number.
  Comparing configurations on the test set leaks it into model selection.

## 1. Best GAP-INR model

[`configs/config_model.yaml`](../configs/config_model.yaml) holds the paper's configuration; train
it unchanged (results vary slightly between runs, see section 0). Key hyperparameters:

| Group | Setting |
|---|---|
| Decoder | SIREN, `hidden_size: 384`, `num_hidden_layers: 7` (8 sine layers), all FiLM-modulated (`modulated_layers: [0..7]`) |
| SIREN frequency | `omega_0 = omega_start = omega_end = 30`, `schedule_type: constant` |
| Latent | `latent_dim: [256, 32, 32]` (256 channels on a 32×32 grid), one per eye |
| Conditioning | FiLM vector of two scalars: weeks-from-baseline (the temporal axis) and age at visit (`faf_ga_twovar_wktemporal_512`; `cond_dims` is derived from the enabled conditions at runtime) |
| Heads / loss | FAF reconstruction (MSE) + GA segmentation (cross-entropy + soft Dice); `sr_weight: 10`, `seg_weight: 1` |
| Optimisation | AdamW, `lr_inr: 1.0e-4`, `lr_latent: 5.0e-3`, weight decay 0.1 on both, cosine schedule to `2.5e-5`, `n_samples: 10000` coordinates per step |
| Schedule | `epochs.train: 50`, validation every 5 epochs holding out each val eye's last visit |
| Test-time optimisation | `epochs.val: 25`, early stopping on the fitted visits' DICE (patience 10), latent L2 `val_latent_reg: 0.01`, segmentation loss on the fitted visits (`seg_loss_val: true`) |
| Data | 620 centre-crop, then resize to a 512 grid (`faf_ga_twovar_wktemporal_512`) |
| Rendering | `renormalize_output: true` (each rendered FAF min-max scaled to [0, 1]), `seg_threshold: 0.5` |

Train and evaluate it:

```bash
python run.py
python evaluate.py --checkpoint runs/faf_ga/<run>/checkpoint_best.pth \
    --holdout_strategy leave_one_out --test on
```

`<run>` is the timestamped directory `run.py` prints as `Output directory:`. It is nested under the
config's `output_dir`: `runs/faf_ga/faf_ga_twovar_wktemporal_512_<timestamp>_loc/` (`loc` becomes the
last three digits of `SLURM_JOB_ID` under SLURM).

`evaluate.py` writes `evaluation_*/leave_one_out_summary.csv` with the held-out DICE, PSNR, SSIM,
LPIPS, Hausdorff distance and lesion-area MAE, per held-out visit position and grouped into
interpolation and extrapolation, for the validation and the test split. The test rows are the GAP-INR
numbers of the results table.

### Forecast scenarios

The forecasting scenarios and the commands that run them are in the README, under
[The workflow](../README.md#the-workflow-train-validate-test-forecast).

## 2. Changing the model

Every architectural knob is documented in `configs/config_model.yaml` and in the README's model
architecture options. Change one at a time, in a copy of the config:

```bash
cp configs/config_model.yaml configs/my_config.yaml   # edit it, then:
python run.py --config_model configs/my_config.yaml
```

A few knobs are also exposed on the command line. These are the only ones `run.py` accepts; anything
else must go in the config. List-valued flags take space-separated integers, not a bracketed string:

```bash
python run.py --inr_decoder__latent_dim 64 32 32   # smaller latent grid
python run.py --inr_decoder__hidden_size 256
python run.py --seed 1927
```

`python run.py --help` lists the full set (`--config_data`, `--config_model`, `--seed`, the
`--inr_decoder__*` shape flags, `--model_gen__cond_scale`, `--n_subjects__*` and the `--overfit*`
flags).

Select on the validation leave-one-out metric, not on test.

## 3. Baselines

The comparison methods are the ImageFlowNet family (ImageFlowNet ODE, time-conditioned U-Net,
time-aware diffusion), evaluated on the same test eyes, trained, rendered and scored at 256x256 with
seed 1 (the paper configuration). The protocol is in [`BASELINES.md`](BASELINES.md); the code is in
[`../baselines/imageflownet/`](../baselines/imageflownet/), under a non-commercial licence.

## 4. Figures

```bash
# Per-eye GA-area progression panel (the paper figure)
python plot_lesion_size_trajectories.py \
    --csv runs/faf_ga/<run>/evaluation_*/lesion_analysis/lesion_areas_test_epoch_*.csv --split test \
    --holdout-dir runs/faf_ga/<run>/evaluation_*/holdout_timeline_arrays
```

