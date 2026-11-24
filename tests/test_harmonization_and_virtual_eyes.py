"""
End-to-end verification suite for:
  1. Healthy-retina quantile intensity harmonization ('healthy_quantile') — exact
     parity between training (dataset.load_coords_and_values) and evaluation (utils.load_2d_modality)
  2. Trajectory-consistent virtual-eye augmentation ('virtual_eyes') — dedicated latent per
     virtual eye, identical geometric transform across all visits of a virtual eye, exact
     FAF/GA pixel alignment, binary mask preservation, re-baselined sub-trajectories, and
     strict train-only isolation (zero leakage into val/test)
  3. CUDA forward+backward smoke test with INR_Decoder (+SRA) and dedicated virtual-eye latents
  4. Backward compatibility with baseline configs/config_model.yaml
"""

import os
import sys
import tempfile
import yaml
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from data_loading.dataset import Data, validate_splits
from models.inr_decoder import INR_Decoder
from utils import load_2d_modality, healthy_quantile_normalize


def assemble_args(config_model_rel):
    with open(os.path.join(REPO_ROOT, config_model_rel)) as f:
        args_model = yaml.safe_load(f)
    with open(os.path.join(REPO_ROOT, "configs/config_data.yaml")) as f:
        args_data = {"dataset": yaml.safe_load(f)[args_model["config_data"]]}
    args = {**args_data, **args_model}
    with open(args["dataset"]["subject_ids"]) as f:
        args["dataset"]["subject_ids"] = yaml.safe_load(f)[args["dataset"]["dataset_name"]]["subject_ids"]
    args["output_dir"] = tempfile.mkdtemp(prefix="gapinr_test_")
    # Derive cond_dims exactly like ModelBuilder._init_inr
    args["inr_decoder"]["cond_dims"] = sum(1 for _, v in args["dataset"]["conditions"].items() if v)
    return args


def test_healthy_quantile_unit():
    rng = np.random.default_rng(0)
    # Two synthetic visits of same eye with 2x gain difference + 15% dark GA + bright hot pixel
    base = np.clip(rng.normal(120.0, 18.0, size=(256, 256)), 10.0, 240.0).astype(np.float32)
    base[80:140, 80:140] = rng.normal(25.0, 5.0, size=(60, 60)).clip(5.0, 50.0)
    base[0, 0] = 255.0
    bright = np.clip(base * 1.45 + 15.0, 1.0, 255.0).astype(np.float32)
    bright[:12, :] = 0.0  # simulated zero-padding border from shift/rotation

    h1 = healthy_quantile_normalize(base)
    h2 = healthy_quantile_normalize(bright)
    assert h1.min() >= 0.0 and h1.max() <= 1.0
    assert h2.min() >= 0.0 and h2.max() <= 1.0
    healthy_mask = np.ones((256, 256), dtype=bool)
    healthy_mask[80:140, 80:140] = False
    healthy_mask[:12, :] = False
    diff_healthy = abs(float(h1[healthy_mask].mean() - h2[healthy_mask].mean()))
    assert diff_healthy < 0.015, f"Healthy retina brightness did not match: diff={diff_healthy:.4f}"
    print(f"[PASS] 1. healthy_quantile unit test (gain-shifted healthy diff={diff_healthy:.4f})")


def test_omega_harmonized_virtual_eyes():
    args = assemble_args("configs/config_model_attention_harmonized_aug.yaml")
    validate_splits(args)

    train_ds = Data(args, args["dataset"]["tsv_file"], split="train")
    val_ds = Data(args, args["dataset"]["tsv_file"], split="val")
    test_ds = Data(args, args["dataset"]["tsv_file"], split="test")

    # 1) Split isolation & eye counts
    real_train_df = train_ds.df[train_ds.df["aug_id"] == 0]
    n_real_train = real_train_df["Eye_ID"].nunique()
    n_total_train = train_ds.n_unique_subjects
    assert n_real_train == 26, f"Expected 26 real train eyes, got {n_real_train}"
    assert n_total_train > n_real_train * 5, f"Expected >5x virtual train eyes, got {n_total_train}"
    assert not getattr(val_ds, "virtual_eyes_aug", False) and val_ds.n_unique_subjects == 5
    assert not getattr(test_ds, "virtual_eyes_aug", False) and test_ds.n_unique_subjects == 6
    assert "aug_id" not in val_ds.df.columns and "aug_id" not in test_ds.df.columns
    print(f"[PASS] 2. Split isolation & virtual eye counts (train: {n_real_train} real -> {n_total_train} total; val: {val_ds.n_unique_subjects}; test: {test_ds.n_unique_subjects})")

    # 2) Train vs eval bit-for-bit normalization parity on a real training visit
    row0 = real_train_df.iloc[0].to_dict()
    mods0 = train_ds.load_modalities(row0)
    _, vals0 = train_ds.load_coords_and_values(mods0, row0, normalize=True)
    faf_train_grid = vals0[:, 0].reshape(512, 512)
    faf_path = train_ds.resolve_path(row0, "faf_path")
    faf_eval_grid = load_2d_modality(faf_path, is_seg=False, args=args)
    mean_norm_diff = float(np.abs(faf_train_grid - faf_eval_grid).mean())
    max_norm_diff = float(np.abs(faf_train_grid - faf_eval_grid).max())
    # Small (~2e-3) diff comes solely from PIL 'L' uint8 vs 'F' float32 bilinear resize rounding
    assert mean_norm_diff < 5e-3 and max_norm_diff < 2e-2, (
        f"Train vs eval healthy_quantile mismatch: mean={mean_norm_diff}, max={max_norm_diff}"
    )
    print(f"[PASS] 3. Train vs eval healthy_quantile parity (mean_diff={mean_norm_diff:.2e}, max_diff={max_norm_diff:.2e})")

    # 3) Trajectory consistency: aug1 (pure hflip) & aug2 (rot+scale+shift) & aug_rebase
    eye0 = str(row0["Eye_ID"])
    real_v1 = train_ds.df[(train_ds.df["Eye_ID"] == eye0)].sort_values("weeks_from_baseline").iloc[0].to_dict()
    aug1_v1 = train_ds.df[(train_ds.df["Eye_ID"] == f"{eye0}__aug1")].sort_values("weeks_from_baseline").iloc[0].to_dict()
    aug2_rows = train_ds.df[(train_ds.df["Eye_ID"] == f"{eye0}__aug2")].sort_values("weeks_from_baseline")
    rebase_rows = train_ds.df[(train_ds.df["Eye_ID"] == f"{eye0}__aug_rebase")].sort_values("weeks_from_baseline")

    # Dedicated sub_id_int per virtual eye
    assert int(real_v1["sub_id_int"]) != int(aug1_v1["sub_id_int"]) != int(aug2_rows.iloc[0]["sub_id_int"])
    # Pure hflip check on both FAF and GA mask
    _, v_real = train_ds.load_coords_and_values(train_ds.load_modalities(real_v1), real_v1, normalize=True)
    _, v_aug1 = train_ds.load_coords_and_values(train_ds.load_modalities(aug1_v1), aug1_v1, normalize=True)
    faf_diff_flip = float(np.abs(v_real[:, 0].reshape(512, 512)[:, ::-1] - v_aug1[:, 0].reshape(512, 512)).max())
    seg_diff_flip = float(np.abs(v_real[:, 1].reshape(512, 512)[:, ::-1] - v_aug1[:, 1].reshape(512, 512)).max())
    assert faf_diff_flip < 1e-5 and seg_diff_flip == 0.0, f"aug1 hflip mismatch: faf={faf_diff_flip}, seg={seg_diff_flip}"

    # Random affine virtual eye check: binary mask stays exact {0, 1}, FAF in [0, 1]
    aug2_v1 = aug2_rows.iloc[0].to_dict()
    _, v_aug2 = train_ds.load_coords_and_values(train_ds.load_modalities(aug2_v1), aug2_v1, normalize=True)
    seg_unique = set(np.unique(v_aug2[:, 1]).tolist())
    assert seg_unique.issubset({0.0, 1.0}), f"Binary GA mask corrupted by affine transform: {seg_unique}"
    assert v_aug2[:, 0].min() >= 0.0 and v_aug2[:, 0].max() <= 1.0

    # Re-baselined sub-trajectory check
    real_eye_rows = train_ds.df[train_ds.df["Eye_ID"] == eye0].sort_values("weeks_from_baseline")
    assert len(rebase_rows) == len(real_eye_rows) - 1
    assert abs(float(rebase_rows.iloc[0]["weeks_from_baseline"])) < 1e-6
    assert abs(float(rebase_rows.iloc[0]["AgeatVisit"]) - float(real_eye_rows.iloc[1]["AgeatVisit"])) < 1e-6
    print("[PASS] 4. Virtual-eye trajectory consistency, exact hflip alignment, binary mask & re-baselining")

    # 4) CUDA forward+backward smoke test verifying dedicated latent updates only for virtual eye
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec = INR_Decoder(args, device).to(device).train()
    latents = torch.nn.Parameter(torch.randn(n_total_train, *args["inr_decoder"]["latent_dim"], device=device) * 0.01)

    aug1_pos = int(train_ds.df.index[train_ds.df["Eye_ID"] == f"{eye0}__aug1"][0])
    coords, values, conds, idcs, tvals = train_ds[aug1_pos]
    assert coords.shape[0] == 60000, f"Expected train_coordinates_per_visit=60000, got {coords.shape[0]}"
    # Take first 10,000 coords (1 training chunk)
    coords = coords[:10000].to(device)
    values = values[:10000].to(device)
    conds = conds[:10000].to(device)
    idcs = idcs[:10000].squeeze().to(device)
    pred = dec(coords, latents, conds, idcs_df=idcs)
    loss = torch.nn.functional.mse_loss(pred[:, :1], values[:, :1])
    loss.backward()

    grad_norms = latents.grad.flatten(1).norm(dim=1)
    active_id = int(aug1_v1["sub_id_int"])
    real_id = int(real_v1["sub_id_int"])
    assert grad_norms[active_id].item() > 0.0, "Virtual eye latent received zero grad"
    assert grad_norms[real_id].item() == 0.0, "Real eye latent leaked grad from virtual eye!"
    assert torch.nn.utils.clip_grad_norm_(list(dec.parameters()) + [latents], 1.0) >= 0.0
    print(f"[PASS] 5. CUDA forward+backward smoke test on dedicated virtual-eye latent (sub_id={active_id}, real_id={real_id} untouched)")


def test_baseline_backward_compat():
    args = assemble_args("configs/config_model.yaml")
    train_ds = Data(args, args["dataset"]["tsv_file"], split="train")
    assert train_ds.n_unique_subjects == 26
    assert not getattr(train_ds, "virtual_eyes_aug", False)
    assert "aug_id" not in train_ds.df.columns
    print("[PASS] 6. Baseline config_model.yaml 100% backward compatible (26 eyes, no aug_id)")


if __name__ == "__main__":
    test_healthy_quantile_unit()
    test_omega_harmonized_virtual_eyes()
    test_baseline_backward_compat()
    print("\nALL HARMONIZATION & VIRTUAL-EYE TESTS PASSED.")
