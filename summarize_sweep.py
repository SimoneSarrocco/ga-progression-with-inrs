"""
Summarize a completed or in-progress GAP-INR Optuna study:
  1. Prints top-K architectures with all held-out VAL_EVAL metrics (DICE, LPIPS, PSNR, SSIM, HD, LOSS)
  2. Computes fANOVA hyperparameter importance ranking (replaces infeasible grid search)
  3. Identifies the DICE vs. LPIPS Pareto-optimal trials
  4. Exports a ready-to-run full 50-epoch YAML config for the best trial
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import yaml
import pandas as pd
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sweep_optuna import suggest_trial_config, load_base_args


def export_best_config(study: optuna.Study, base_config_path: str, config_data: str | None, out_yaml: str):
    """Materialize a clean full 50-epoch runnable config_model YAML for the winning trial."""
    best_t = study.best_trial
    base_args = load_base_args(base_config_path, config_data)
    fixed = optuna.trial.FixedTrial(best_t.params)
    resolved = suggest_trial_config(fixed, base_args, max_epochs=50)

    # Restore full paper training/evaluation flags for final retraining
    with open(base_config_path, "r") as f:
        orig_model = yaml.safe_load(f)
    out = copy.deepcopy(orig_model)
    out["output_dir"] = f"./runs/{study.study_name}_best_trial{best_t.number}"
    out["inr_decoder"] = copy.deepcopy(resolved["inr_decoder"])
    out["data_augmentation"] = copy.deepcopy(resolved["data_augmentation"])
    out["optimizer"]["val_latent_init"] = resolved["optimizer"]["val_latent_init"]
    out["optimizer"]["val_latent_reg"] = resolved["optimizer"]["val_latent_reg"]
    out["optimizer"]["max_grad_norm"] = resolved["optimizer"]["max_grad_norm"]
    out["epochs"]["train"] = 50
    out["test"] = {"activate": True}
    out["save_model"] = True
    out["save_imgs"] = {"train": False, "val": True, "test": False}
    out["lesion_analysis"] = {
        "activate": True, "save_individual_plots": False,
        "save_individual_plots_final": True, "max_individual_plots": 99999,
    }
    if config_data:
        out["config_data"] = config_data

    with open(out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"Exported full 50-epoch config for best trial #{best_t.number} -> {out_yaml}")


def main():
    parser = argparse.ArgumentParser(description="Analyze a GAP-INR Optuna sweep study")
    parser.add_argument("--study", type=str, default="gapinr_omega_arch")
    parser.add_argument("--sweep_root", type=str, default=os.path.join(REPO_ROOT, "runs", "optuna_sweeps"))
    parser.add_argument("--base_config", type=str, default=os.path.join(REPO_ROOT, "configs", "config_model_attention_harmonized_aug.yaml"))
    parser.add_argument("--config_data", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--export_best", type=str, default=None, help="Optional output YAML path for best trial")
    cmd = parser.parse_args()

    sweep_dir = os.path.join(cmd.sweep_root, cmd.study)
    journal_path = os.path.join(sweep_dir, "journal.log")
    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.load_study(study_name=cmd.study, storage=storage)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    print(f"Study '{cmd.study}': {len(study.trials)} total | {len(complete)} complete | {len(pruned)} pruned | {len(running)} running")

    if not complete:
        print("No completed trials yet.")
        return

    rows = []
    for t in complete:
        ua = t.user_attrs
        rows.append({
            "trial": t.number,
            "obj": round(float(t.value), 4),
            "DICE": round(float(ua.get("best_dice", float("nan"))), 4),
            "LPIPS": round(float(ua.get("best_lpips", float("nan"))), 4),
            "PSNR": round(float(ua.get("best_psnr", float("nan"))), 2),
            "SSIM": round(float(ua.get("best_ssim", float("nan"))), 3),
            "HD": round(float(ua.get("best_hd", float("nan"))), 1),
            "ep": int(ua.get("best_epoch", -1)),
            "params_M": round(float(ua.get("n_params", 0)) / 1e6, 2),
            "lat": f"1D-{t.params.get('latent_c_1d')}" if t.params.get("latent_type") == "vec_1d"
                   else f"{t.params.get('latent_c_3d')}x{t.params.get('latent_grid')}^2",
            "siren": f"{t.params.get('hidden_size')}x{t.params.get('num_hidden_layers')}",
            "omega": f"{t.params.get('omega_0')}->{t.params.get('omega_end', t.params.get('omega_0'))}",
            "SRA": t.params.get("sra_activate", False),
            "CRA": t.params.get("cra_activate", False),
            "head": t.params.get("output_head"),
            "cond": t.params.get("cond_encoding"),
            "aug": t.params.get("aug_mode"),
        })
    df = pd.DataFrame(rows).sort_values("obj", ascending=False)
    print(f"\n=== TOP {min(cmd.top_k, len(df))} ARCHITECTURES (Held-out VAL_EVAL) ===")
    print(df.head(cmd.top_k).to_string(index=False))

    # Pareto frontier on (DICE maximize, LPIPS minimize)
    pareto = []
    valid = df.dropna(subset=["DICE", "LPIPS"])
    for _, r in valid.iterrows():
        dominated = ((valid["DICE"] >= r["DICE"]) & (valid["LPIPS"] <= r["LPIPS"]) &
                     ((valid["DICE"] > r["DICE"]) | (valid["LPIPS"] < r["LPIPS"]))).any()
        if not dominated:
            pareto.append(r)
    if pareto:
        print("\n=== PARETO FRONT (DICE vs. LPIPS) ===")
        print(pd.DataFrame(pareto)[["trial", "DICE", "LPIPS", "PSNR", "HD", "params_M", "lat", "siren", "SRA", "CRA", "head", "cond", "aug"]].to_string(index=False))

    # fANOVA parameter importance (needs >= 4 completed trials)
    if len(complete) >= 4:
        try:
            imp = optuna.importance.get_param_importances(study)
            print("\n=== HYPERPARAMETER IMPORTANCE (fANOVA) ===")
            for k, v in imp.items():
                print(f"  {k:28s}: {100.0 * v:5.1f}%")
        except Exception as e:
            print(f"\n(Could not compute fANOVA importance yet: {e})")

    out_yaml = cmd.export_best or os.path.join(sweep_dir, "config_model_optuna_best.yaml")
    export_best_config(study, cmd.base_config, cmd.config_data, out_yaml)


if __name__ == "__main__":
    main()
