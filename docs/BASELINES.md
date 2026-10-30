# Comparison baselines

The paper compares GAP-INR against the **ImageFlowNet** model family. The code is included, in
[`baselines/imageflownet/`](../baselines/imageflownet/), together with the run commands and the
protocol: see its [README](../baselines/imageflownet/README.md) to reproduce the baseline numbers,
and its [ATTRIBUTION.md](../baselines/imageflownet/ATTRIBUTION.md) for provenance and licensing.

> That directory is a derivative of [ImageFlowNet](https://github.com/ChenLiu-1996/ImageFlowNet) and
> is **not** covered by this repository's Apache-2.0 licence. It is governed by the Yale
> Non-Commercial licence shipped alongside it. See the "Scope of this licence" note at the end of the
> top-level [`LICENSE`](../LICENSE).

## Methods

| Paper name | `--model` flag | What it is |
|---|---|---|
| **ImageFlowNet (ODE)** | `ImageFlowNetODE` | neural-ODE latent forecaster |
| **Time-conditioned U-Net** (T-UNet) | `T_UNet` | guided-diffusion U-Net, sinusoidal time embedding + FiLM |
| **Time-aware diffusion** (T-I2SBUNet) | `I2SBUNet` | Image-to-Image Schrödinger Bridge diffusion |

All three share one harness. T-UNet and T-I2SBUNet are the two comparison methods used by the
ImageFlowNet paper itself.

## What the comparison shares

- **Test eyes.** Both the baselines and GAP-INR read the `split` column of the same clinical CSV (see
  [`DATA.md`](DATA.md)). GAP-INR checks its resolved eyes against
  [`configs/expected_split.yaml`](../configs/expected_split.yaml); the baselines' evaluation
  hard-asserts that the eyes it scores are exactly the six test eyes of
  `baselines/imageflownet/eval_spec.py`, which are GAP-INR's test eyes.
- **Training eyes.** `eval_spec.py` counts only longitudinal eyes, meaning at least two visits whose
  FAF and grader-1 mask files are both on disk: 23 train eyes. `expected_split.yaml` counts every
  training eye with a FAF path: 26 train eyes, because GAP-INR also trains on eyes with a single usable
  visit and its majority vote can use grader 2 or 3 when grader 1 is missing. Validation (5) and
  test (6) are identical under both.
- **Geometry and intensity.** The same 620 centre-crop of the native image, then a resize (FAF bicubic
  in the baselines, bilinear in GAP-INR; masks nearest in both), and the same per-visit min-max
  normalisation.
- **Scoring grid.** The paper runs the baselines at `--target-dim (256,256)`, so they are trained,
  rendered and scored at 256x256 (`METRIC_DIM` equals the model grid unless `--metric-resize` is
  given). Hausdorff distances are then in 256-grid pixels. To score GAP-INR on the same grid, evaluate
  it with `--config_data faf_ga_twovar_wktemporal_512_to256`, which renders at 512 and resizes
  prediction and ground truth to 256 before scoring.

## Protocol

The ImageFlowNet family outputs images only, so GA DICE is computed by passing the predicted FAF
through a single frozen segmentor, trained once and reused for all three forecasters. GA is scored
against the real GA masks, not against the segmentor's reading of the ground-truth image.

```bash
cd baselines/imageflownet/src
SEG='$ROOT/checkpoints/segment_retina_faf_ga_512_seed1.pty'   # $ROOT is expanded by the scripts

# 0. shared segmentor (train once)
python train_segmentor.py --dataset-name retina_faf_ga --target-dim '(256,256)' --segmentor-ckpt "$SEG"

# 1-3. train each forecaster
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model ImageFlowNetODE --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model T_UNet        --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model I2SBUNet --diffusion-interval 100 --segmentor-ckpt "$SEG"

# 4. evaluation (paper numbers): PSNR/SSIM/LPIPS + DICE(segmentor(pred FAF), real GA mask)
python eval_faf_ga.py --model ImageFlowNetODE --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_faf_ga.py --model T_UNet          --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_faf_ga.py --model I2SBUNet        --target-dim '(256,256)' --diffusion-interval 100 --segmentor-ckpt "$SEG"
```

The segmentor checkpoint keeps its historical name (`..._512_...`) although it is trained at 256 here.

### Paper configuration

- **Resolution 256x256, seed 1.**
- `--diffusion-interval` (I2SBUNet only, default 100) is the number of diffusion steps and must
  match between training and evaluation. T-UNet and ImageFlowNetODE ignore it.
- **Scenario 1** is a single-pair forecast: for every pair of visits (a, b) with a before b, visit b
  is predicted from visit a alone. Visit 1 is never a target. A pair counts as extrapolation when b is
  the eye's last visit and as interpolation otherwise, so on the 6 test eyes there are 36 pairs, 18 in
  each bucket. GAP-INR's `evaluate.py --pairwise` runs the same protocol.
- **Scenario 2** uses the full patient history through test-time optimisation. T-I2SBUNet has no
  test-time optimisation, so scenario 2 does not apply to it.

Each run writes `leave_one_out_summary_test_<best_type>.csv` (`..._seg_dice.csv` with the default
`--best-type seg_dice`), plus `..._copyforward_<best_type>.csv` for the copy-forward reference
(predict the source visit unchanged) and, for scenario 2, `..._tto_<best_type>.csv`. These files
report the same metrics as GAP-INR's [`summarise_eval.py`](../summarise_eval.py) (DICE, PSNR, SSIM,
LPIPS, Hausdorff distance, lesion-area MAE, interpolation vs extrapolation, and minor/major GA
growth), but in their own layout (`Set`, `<metric>_mean`, `<metric>_std`, `n`) and pooled over pairs,
whereas `summarise_eval.py` reports the mean and standard error over eyes.

Classical non-learned references live in `baselines/imageflownet/comparison/interpolation/`:
per-pixel linear, linear-regression and cubic-spline interpolation/extrapolation, GA-area growth-rate
extrapolation, and copy-forward.
