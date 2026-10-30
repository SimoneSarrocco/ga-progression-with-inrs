# ImageFlowNet baselines

The comparison methods reported in the paper, adapted from [ImageFlowNet](https://github.com/ChenLiu-1996/ImageFlowNet)
to the same cohort and test eyes as GAP-INR. Provenance, the changes made for this paper, and the
licensing of this directory are in [ATTRIBUTION.md](ATTRIBUTION.md).

> **Licence.** This directory is **not** covered by the repository's Apache-2.0 licence. It is a
> derivative of ImageFlowNet and is governed by the Yale Non-Commercial licence in
> [LICENSE.md](LICENSE.md); the vendored `external_src/I2SB/` carries NVIDIA's own non-commercial
> licence. Research and evaluation use only.

## Methods

| Paper name | `--model` | What it is |
|---|---|---|
| ImageFlowNet (ODE) | `ImageFlowNetODE` | neural-ODE latent forecaster |
| Time-conditioned U-Net (T-UNet) | `T_UNet` | guided-diffusion U-Net, sinusoidal time embedding + FiLM |
| Time-aware diffusion (T-I2SBUNet) | `I2SBUNet` | Image-to-Image Schrödinger Bridge diffusion |

All three share one harness. T-UNet and T-I2SBUNet are the two comparison methods used by the
ImageFlowNet paper itself.

## What is shared with GAP-INR

These models output images only, so GA segmentation comes from a single frozen segmentor trained once
and reused for all three forecasters. The rest follows GAP-INR:

- `eval_spec.py` pins the canonical eye lists, and the evaluation hard-asserts that the scored test
  eyes are GAP-INR's six test eyes.
- `common_preproc.py` applies the same per-visit min-max normalisation as GAP-INR.
- The loader centre-crops the native 768 px image to 620 px (a direct 512 crop would clip GA) and
  resizes it to `--target-dim` (FAF bicubic, mask nearest); predictions and ground truth are scored on
  that grid, 256x256 in the paper.

`eval_spec.py` and the loader read the repository's `data/clinical_metadata.csv` (`GAPINR_CSV`
overrides the path), the table GAP-INR reads. See [`../../docs/DATA.md`](../../docs/DATA.md).

## Install

```bash
pip install -r ../../requirements.txt -r requirements.txt
```

## Run

All commands are run from `src/`. The scripts expand `$ROOT` to this directory, hence the single
quotes.

```bash
cd baselines/imageflownet/src
SEG='$ROOT/checkpoints/segment_retina_faf_ga_512_seed1.pty'

# 0. shared segmentor (train once, reused by all three forecasters)
python train_segmentor.py --dataset-name retina_faf_ga --target-dim '(256,256)' --segmentor-ckpt "$SEG"

# 1-3. train each forecaster
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model ImageFlowNetODE --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model T_UNet         --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model I2SBUNet --diffusion-interval 100 --segmentor-ckpt "$SEG"

# 4. evaluate (the paper numbers)
python eval_faf_ga.py --model ImageFlowNetODE --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_faf_ga.py --model T_UNet          --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_faf_ga.py --model I2SBUNet        --target-dim '(256,256)' --diffusion-interval 100 --segmentor-ckpt "$SEG"
```

Each evaluation scores every forward visit pair (a, b) of the test eyes, predicting visit b from
visit a, and writes `leave_one_out_summary_test_<best_type>.csv` into the run directory
(`..._seg_dice.csv` with the default `--best-type`): DICE, PSNR, SSIM, LPIPS, Hausdorff distance and
lesion-area MAE, split into interpolation (b is not the eye's last visit) and extrapolation (it is).
These are the metrics of GAP-INR's [`summarise_eval.py`](../../summarise_eval.py), pooled over pairs
with a standard deviation rather than averaged per eye. A copy-forward reference (predict the source
visit unchanged) is written to `..._copyforward_<best_type>.csv` as the floor any model has to beat.

### Paper configuration

- **Resolution 256×256, seed 1.**
- `--diffusion-interval` applies to `I2SBUNet` only and **must match** between training and
  evaluation; the other two models ignore it.
- **Scenario 1** is a single-pair forecast (an older visit predicts a newer one); **scenario 2** uses the full
  patient history via test-time adaptation. T-I2SBUNet has no test-time-adaptation mechanism, so
  scenario 2 does not apply to it.

## Classical floors

`comparison/interpolation/` scores non-learned references: per-pixel linear, linear-regression and
cubic-spline interpolation/extrapolation, GA-area growth-rate extrapolation, and copy-forward.
`run_seg_interp.py` interpolates the GA masks directly, so it needs no GPU and no segmentor (numpy,
scipy, pandas, Pillow and scikit-image).

## Layout

```
eval_spec.py        canonical eye lists (same val/test eyes as GAP-INR)
common_preproc.py   per-visit intensity normalisation (same as GAP-INR)
monitor_panel.py    shared TensorBoard validation panel
src/
  train_segmentor.py    trains the shared frozen GA segmentor
  train_2pt_all.py      trains a model (--model selects which)
  eval_faf_ga.py        pairwise forecast evaluation against the real GA masks
  datasets/             the cohort loader
  nn/                   model implementations
  utils/, data_utils/   upstream helpers
external_src/I2SB/  I2SB + guided-diffusion (see ATTRIBUTION.md)
comparison/         classical interpolation/extrapolation floors
```
