"""
Multi-GPU, multi-fidelity Optuna architecture & augmentation ablation harness for GAP-INR.

Searches jointly over:
  - SIREN width & depth + FiLM modulation scope
  - Latent geometry: 3D spatial grid [C, H, W] vs. 1D global vector [C, 1, 1]
  - SIREN omega schedule (constant vs. linear decay)
  - Attention mechanisms: scale-preserving LatentSpatialAttention (+SRA) & ChannelRecalibration (+CRA)
  - Output head: separated linear (default), shared single layer, separated MLP, dedicated seg branch
  - FiLM conditioning encoding & embedding dim: raw (no encoding), Fourier, joint MLP, separate MLP
  - Augmentation: none vs. trajectory-consistent virtual_eyes (copies, rotation, re-baselining)
  - TTO prior: val_latent_init & val_latent_reg

Uses lock-free Optuna JournalStorage so multiple A100 workers can share one study safely:
  for gpu in 0 1 3 4 5 7; do
      CUDA_VISIBLE_DEVICES=$gpu python sweep_optuna.py --study gapinr_arch --n_trials 12 &
  done

Strict zero test leakage: test.activate is forced False during every sweep trial; all pruning
and selection happen exclusively on held-out VAL_EVAL metrics.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import shutil
import sys
import traceback
import yaml
import numpy as np
import torch
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from build_model import ModelBuilder
from data_loading.dataset import validate_splits
from utils import validate_modality_config


def compute_objective(summary: dict, mode: str) -> float:
    """Compute scalar validation objective (always MAXIMIZED in Optuna)."""
    dice = summary.get("DICE", float("nan"))
    lpips = summary.get("LPIPS", float("nan"))
    loss = summary.get("LOSS", float("nan"))

    if mode == "composite":
        if math.isnan(dice) or math.isnan(lpips):
            return float("-inf")
        return float(dice - 0.5 * lpips)
    if mode == "dice":
        return float(dice) if not math.isnan(dice) else float("-inf")
    if mode == "neg_loss":
        return float(-loss) if not math.isnan(loss) else float("-inf")
    raise ValueError(f"Unknown objective mode: {mode}")


def suggest_trial_config(trial: optuna.Trial, base_args: dict, max_epochs: int) -> dict:
    """Sample a full, valid GAP-INR config dictionary from the conditional Optuna search space."""
    args = copy.deepcopy(base_args)
    inr = args["inr_decoder"]
    opt = args["optimizer"]
    ds = args["dataset"]

    # ---- 1. SIREN width, depth & FiLM modulation scope ----
    hidden_size = trial.suggest_categorical("hidden_size", [256, 384, 512])
    num_hidden = trial.suggest_int("num_hidden_layers", 4, 8)
    mod_scope = trial.suggest_categorical("mod_scope", ["all", "last_half"])
    inr["hidden_size"] = int(hidden_size)
    inr["num_hidden_layers"] = int(num_hidden)
    if mod_scope == "all":
        inr["modulated_layers"] = list(range(num_hidden + 1))
    else:
        start = max(1, (num_hidden + 1) // 2)
        inr["modulated_layers"] = list(range(start, num_hidden + 1))

    # ---- 2. Latent geometry: 3D spatial grid [C, H, W] vs. 1D global vector [C, 1, 1] ----
    latent_type = trial.suggest_categorical("latent_type", ["grid_3d", "vec_1d"])
    if latent_type == "vec_1d":
        c_1d = trial.suggest_categorical("latent_c_1d", [128, 256, 512])
        inr["latent_dim"] = [int(c_1d), 1, 1]
    else:
        grid_sz = trial.suggest_categorical("latent_grid", [8, 16, 32])
        c_3d = trial.suggest_categorical("latent_c_3d", [64, 128, 256])
        inr["latent_dim"] = [int(c_3d), int(grid_sz), int(grid_sz)]

    # ---- 3. SIREN omega schedule ----
    omega_sched = trial.suggest_categorical("omega_schedule", ["constant", "linear"])
    omega_0 = trial.suggest_categorical("omega_0", [15, 20, 30, 45])
    inr["schedule_type"] = omega_sched
    inr["omega_0"] = int(omega_0)
    inr["omega_start"] = int(omega_0)
    if omega_sched == "constant":
        inr["omega_end"] = int(omega_0)
    else:
        omega_end = trial.suggest_categorical("omega_end", [5, 10, 15, 20])
        inr["omega_end"] = int(min(omega_end, omega_0))

    # ---- 4. Attention mechanisms (SRA on 3D grid only; CRA on any latent type) ----
    if latent_type == "grid_3d":
        sra_on = trial.suggest_categorical("sra_activate", [False, True])
    else:
        sra_on = False
    if sra_on:
        sra_layers = trial.suggest_categorical("sra_layers", [1, 2])
        sra_heads = trial.suggest_categorical("sra_heads", [4, 8])
        sra_res_scale = trial.suggest_categorical("sra_residual_scale", [0.2, 0.35, 0.5])
        inr["latent_attention"] = {
            "activate": True,
            "num_layers": int(sra_layers),
            "num_heads": int(sra_heads),
            "ffn_mult": 2.0,
            "dropout": 0.0,
            "use_ffn": False,
            "use_pos_embed": True,
            "residual_scale": float(sra_res_scale),
        }
    else:
        inr["latent_attention"] = {"activate": False}

    cra_on = trial.suggest_categorical("cra_activate", [False, True])
    if cra_on:
        cra_delta = trial.suggest_categorical("cra_max_delta", [0.1, 0.2])
        inr["channel_recalibration"] = {
            "activate": True,
            "reduction": 8,
            "max_delta": float(cra_delta),
        }
    else:
        inr["channel_recalibration"] = {"activate": False}

    # ---- 5. Output head architecture ----
    output_head = trial.suggest_categorical(
        "output_head", ["separated_linear", "shared_single", "separated_mlp", "seg_branch"]
    )
    inr["shared_output_layer"] = False
    inr["seg_head_num_layers"] = 0
    inr["seg_head_hidden_size"] = 256
    inr["seg_head_use_last_features"] = True
    inr["seg_branch"] = {
        "activate": False, "num_layers": 1, "hidden_size": 256, "branch_layer": -1, "modulate": True
    }
    if output_head == "shared_single":
        inr["shared_output_layer"] = True
    elif output_head == "separated_linear":
        inr["seg_head_use_last_features"] = bool(
            trial.suggest_categorical("seg_head_use_last_features", [True, False])
        )
    elif output_head == "separated_mlp":
        inr["seg_head_num_layers"] = int(trial.suggest_categorical("seg_head_num_layers", [1, 2]))
        inr["seg_head_hidden_size"] = int(trial.suggest_categorical("seg_head_hidden_size", [128, 256]))
    elif output_head == "seg_branch":
        inr["seg_branch"]["activate"] = True
        inr["seg_branch"]["num_layers"] = int(trial.suggest_categorical("seg_branch_num_layers", [1, 2]))
        inr["seg_branch"]["hidden_size"] = int(trial.suggest_categorical("seg_branch_hidden_size", [256, 384]))

    # ---- 6. FiLM conditioning encoding & embedding dimension ----
    cond_enc = trial.suggest_categorical("cond_encoding", ["raw", "fourier", "mlp", "mlp_separate"])
    inr["cond_encoding"] = cond_enc
    if cond_enc == "raw":
        inr["cond_num_frequencies"] = 0
    elif cond_enc == "fourier":
        inr["cond_num_frequencies"] = int(trial.suggest_categorical("cond_num_frequencies", [2, 4, 6, 8]))
    elif cond_enc == "mlp":
        inr["cond_num_frequencies"] = 0
        inr["cond_mlp_out"] = int(trial.suggest_categorical("cond_mlp_out", [8, 16, 32, 64]))
        inr["cond_mlp_hidden"] = int(trial.suggest_categorical("cond_mlp_hidden", [16, 32, 64]))
        inr["cond_mlp_layers"] = int(trial.suggest_categorical("cond_mlp_layers", [2, 3]))
    elif cond_enc == "mlp_separate":
        inr["cond_num_frequencies"] = 0
        inr["cond_mlp_out"] = int(trial.suggest_categorical("cond_mlp_out_sep", [4, 8, 16, 32]))
        inr["cond_mlp_hidden"] = int(trial.suggest_categorical("cond_mlp_hidden_sep", [16, 32]))
        inr["cond_mlp_layers"] = int(trial.suggest_categorical("cond_mlp_layers_sep", [2, 3]))

    # ---- 7. Data augmentation choices ----
    aug_mode = trial.suggest_categorical("aug_mode", ["none", "virtual_eyes"])
    if aug_mode == "none":
        args["data_augmentation"] = {"activate": False, "mode": "stochastic"}
        # Unaugmented: use all pixels per visit so per-epoch compute matches augmented trials
        ds["train_coordinates_per_visit"] = 0
    else:
        n_copies = int(trial.suggest_categorical("aug_n_copies", [2, 4, 6]))
        max_rot = float(trial.suggest_categorical("aug_max_rot", [0.0, 10.0, 15.0]))
        rebaseline = bool(trial.suggest_categorical("aug_rebaseline", [False, True]))
        args["data_augmentation"] = {
            "activate": True,
            "mode": "virtual_eyes",
            "virtual_eyes": {
                "n_copies": n_copies,
                "hflip_first": True,
                "hflip_p": 0.5,
                "vflip_p": 0.3,
                "max_rot_deg": max_rot,
                "max_shift_frac": 0.05,
                "scale_range": [0.95, 1.05],
                "rebaseline_subtrajectories": rebaseline,
            },
        }
        # Keep per-epoch step budget roughly constant across n_copies for fair Hyperband pruning
        eff_mult = 1 + n_copies + (1 if rebaseline else 0)
        ds["train_coordinates_per_visit"] = int(max(40000, min(120000, round(260000 / eff_mult / 20000) * 20000)))

    # ---- 8. TTO prior & stability knobs ----
    opt["val_latent_init"] = trial.suggest_categorical("val_latent_init", ["random", "population_mean"])
    opt["val_latent_reg"] = float(trial.suggest_float("val_latent_reg", 0.005, 0.1, log=True))
    opt["max_grad_norm"] = 1.0

    # ---- Sweep runtime hygiene (fast, clean, strictly zero test leakage) ----
    args["epochs"]["train"] = int(max_epochs)
    args["validate_every"] = 5
    args["logging"] = False
    args["tb_writer"] = None
    args["save_model"] = False
    args["save_imgs"] = {"train": False, "val": False, "test": False}
    args["test"] = {"activate": False}
    args["lesion_analysis"] = {
        "activate": False, "save_individual_plots": False,
        "save_individual_plots_final": False, "max_individual_plots": 0,
    }
    args["validation"]["activate"] = True
    args["validation"]["train_eval_every"] = 9999   # skip dense train-eye renders during HPO
    args["validation"]["log_images_every"] = 9999
    args["validation"]["holdout_strategy"] = "last"

    return args


def load_base_args(config_model_path: str, config_data_override: str | None) -> dict:
    with open(config_model_path, "r") as f:
        args_model = yaml.safe_load(f)
    with open(os.path.join(REPO_ROOT, "configs", "config_data.yaml"), "r") as f:
        all_data = yaml.safe_load(f)
    cfg_data_name = config_data_override or args_model.get("config_data", "faf_ga_twovar_wktemporal_512_harmonized")
    args_model["config_data"] = cfg_data_name
    args_data = {"dataset": copy.deepcopy(all_data[cfg_data_name])}
    args = {**args_data, **args_model}
    with open(args["dataset"]["subject_ids"], "r") as f:
        args["dataset"]["subject_ids"] = yaml.safe_load(f)[args["dataset"]["dataset_name"]]["subject_ids"]
    validate_modality_config(args)
    validate_splits(args)
    return args


def enqueue_reference_baseline(study: optuna.Study):
    """Enqueue Trial 0 as the paper-style reference anchor if the study is brand new."""
    if len(study.trials) > 0:
        return
    study.enqueue_trial({
        "hidden_size": 384,
        "num_hidden_layers": 7,
        "mod_scope": "all",
        "latent_type": "grid_3d",
        "latent_grid": 32,
        "latent_c_3d": 256,
        "omega_schedule": "constant",
        "omega_0": 30,
        "sra_activate": False,
        "cra_activate": False,
        "output_head": "separated_linear",
        "seg_head_use_last_features": True,
        "cond_encoding": "raw",
        "aug_mode": "virtual_eyes",
        "aug_n_copies": 4,
        "aug_max_rot": 15.0,
        "aug_rebaseline": True,
        "val_latent_init": "random",
        "val_latent_reg": 0.01,
    })


def export_summary_csv(study: optuna.Study, sweep_dir: str):
    """Write a clean CSV of all trials so progress can be inspected anytime."""
    import pandas as pd
    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "state": t.state.name,
            "objective": t.value,
            **{f"m_{k}": v for k, v in t.user_attrs.items()},
            **t.params,
        }
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows).sort_values("objective", ascending=False, na_position="last")
        df.to_csv(os.path.join(sweep_dir, "trials_summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Optuna architecture & augmentation sweep for GAP-INR")
    parser.add_argument("--study", type=str, default="gapinr_omega_arch", help="Optuna study name")
    parser.add_argument("--sweep_root", type=str, default=os.path.join(REPO_ROOT, "runs", "optuna_sweeps"))
    parser.add_argument("--base_config", type=str, default=os.path.join(REPO_ROOT, "configs", "config_model_attention_harmonized_aug.yaml"))
    parser.add_argument("--config_data", type=str, default=None, help="Optional config_data.yaml section override")
    parser.add_argument("--n_trials", type=int, default=15, help="Trials to run in this worker process")
    parser.add_argument("--max_epochs", type=int, default=30, help="Max train epochs per trial (default 30)")
    parser.add_argument("--min_epochs", type=int, default=5, help="Hyperband min_resource (first val epoch = 5)")
    parser.add_argument("--objective", type=str, default="composite", choices=["composite", "dice", "neg_loss"],
                        help="composite = DICE - 0.5*LPIPS (default), dice = VAL_EVAL DICE, neg_loss = -VAL_EVAL LOSS")
    parser.add_argument("--keep_trial_dirs", action="store_true", help="Keep per-trial output dirs after trial finishes")
    parser.add_argument("--seed", type=int, default=1927)
    cmd = parser.parse_args()

    os.chdir(REPO_ROOT)
    sweep_dir = os.path.join(cmd.sweep_root, cmd.study)
    os.makedirs(sweep_dir, exist_ok=True)
    journal_path = os.path.join(sweep_dir, "journal.log")

    storage = JournalStorage(JournalFileBackend(journal_path))
    sampler = optuna.samplers.TPESampler(seed=cmd.seed, multivariate=True, group=True)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=cmd.min_epochs, max_resource=cmd.max_epochs, reduction_factor=3
    )

    study = optuna.create_study(
        study_name=cmd.study,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    enqueue_reference_baseline(study)
    base_args = load_base_args(cmd.base_config, cmd.config_data)

    def objective_fn(trial: optuna.Trial) -> float:
        trial_dir = os.path.join(sweep_dir, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        args = suggest_trial_config(trial, base_args, max_epochs=cmd.max_epochs)
        args["output_dir"] = trial_dir

        # Save resolved trial config (without non-serializable callback)
        with open(os.path.join(trial_dir, "config_trial.yaml"), "w") as f:
            yaml.safe_dump({k: v for k, v in args.items() if not k.startswith("_") and k != "tb_writer"}, f)

        best = {"obj": float("-inf"), "summary": None}

        def epoch_callback(epoch_idx: int, summary: dict):
            score = compute_objective(summary, cmd.objective)
            step = epoch_idx + 1  # 5, 10, 15, ...
            if score > best["obj"]:
                best["obj"] = score
                best["summary"] = dict(summary)
            trial.report(score, step=step)
            if math.isnan(score) or math.isinf(score):
                raise optuna.TrialPruned(f"Non-finite validation score at epoch {step}")
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {step} (score={score:.4f})")

        args["_epoch_callback"] = epoch_callback

        try:
            builder = ModelBuilder(args)
            n_params = int(sum(p.numel() for p in builder.inr_decoder["train"].parameters()))
            trial.set_user_attr("n_params", n_params)
            if best["summary"] is not None:
                for k, v in best["summary"].items():
                    trial.set_user_attr(f"best_{k.lower()}", float(v) if isinstance(v, (int, float)) else v)
            return float(best["obj"])
        except optuna.TrialPruned:
            if best["summary"] is not None:
                for k, v in best["summary"].items():
                    trial.set_user_attr(f"best_{k.lower()}", float(v) if isinstance(v, (int, float)) else v)
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                trial.set_user_attr("error", "CUDA_OOM")
                torch.cuda.empty_cache()
                raise optuna.TrialPruned("CUDA OOM")
            traceback.print_exc()
            raise
        finally:
            if not cmd.keep_trial_dirs and os.path.isdir(trial_dir):
                shutil.rmtree(trial_dir, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def after_trial(st: optuna.Study, _t: optuna.trial.FrozenTrial):
        try:
            export_summary_csv(st, sweep_dir)
        except Exception:
            pass

    study.optimize(objective_fn, n_trials=cmd.n_trials, callbacks=[after_trial], gc_after_trial=True)
    export_summary_csv(study, sweep_dir)
    print(f"\n[sweep_optuna] Worker done. Summary CSV: {os.path.join(sweep_dir, 'trials_summary.csv')}")


if __name__ == "__main__":
    main()
