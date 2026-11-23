"""
Automated verification suite for INRAM-inspired attention modules in GAP-INR:
  1. Exact zero-init identity (2D & 3D) against baseline GAP-INR
  2. Active-subject slicing & 2-step gradient flow (all attention params wake up)
  3. TTO spatial gradient propagation across the latent grid
  4. Chunked inference() exact equivalence & single grid-attention call per image
  5. Strict backward compatibility with original YAML configs & new attention configs
  6. Parameter count & latency report across the 4-way ablation
"""

import copy
import os
import sys
import time
import yaml
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.inr_decoder import INR_Decoder
from models.attention import LatentSpatialAttention, ChannelRecalibration


def load_yaml(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), "r") as f:
        return yaml.safe_load(f)


def make_minimal_args(
    latent_dim=(256, 32, 32),
    out_dim=(1, 0),
    cond_dims=2,
    latent_attention=None,
    channel_recalibration=None,
    cnn_kernel_size=0,
    hidden_size=384,
    num_hidden_layers=7,
):
    return {
        "device": "cpu",
        "mask_reconstruction": False,
        "inr_decoder": {
            "architecture": "siren",
            "tf_dim": 0,
            "cnn_kernel_size": cnn_kernel_size,
            "latent_dim": list(latent_dim),
            "latent_attention": latent_attention if latent_attention is not None else {"activate": False},
            "channel_recalibration": channel_recalibration if channel_recalibration is not None else {"activate": False},
            "in_dim": len(latent_dim) - 1,
            "cond_dims": cond_dims,
            "cond_encoding": "raw",
            "cond_num_frequencies": 0,
            "time_as_input": False,
            "out_dim": list(out_dim),
            "renormalize_output": False,
            "shared_output_layer": False,
            "seg_head_num_layers": 0,
            "seg_head_hidden_size": 256,
            "seg_head_use_last_features": True,
            "seg_branch": {"activate": False},
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "modulated_layers": list(range(num_hidden_layers + 1)),
            "omega_0": 30,
            "omega_start": 30,
            "omega_end": 30,
            "schedule_type": "constant",
        },
        "encoding": {"activate": False, "type": "none"},
    }


def copy_shared_weights(src_model, dst_model):
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
    dst_model.load_state_dict(dst_sd)


def test_zero_init_identity(device):
    torch.manual_seed(42)
    for ndim, latent_dim in [(2, (64, 16, 16)), (3, (32, 6, 6, 6))]:
        base_args = make_minimal_args(latent_dim=latent_dim, hidden_size=64, num_hidden_layers=3)
        attn_args = make_minimal_args(
            latent_dim=latent_dim,
            hidden_size=64,
            num_hidden_layers=3,
            latent_attention={"activate": True, "num_layers": 2, "num_heads": 4, "ffn_mult": 2.0, "use_ffn": True},
            channel_recalibration={"activate": True, "reduction": 4},
        )
        base_dec = INR_Decoder(base_args, device).to(device).eval()
        attn_dec = INR_Decoder(attn_args, device).to(device).eval()
        copy_shared_weights(base_dec, attn_dec)

        coords = torch.empty(512, ndim, device=device).uniform_(-1, 1)
        latents = torch.randn(4, *latent_dim, device=device)
        conds = torch.randn(512, 2, device=device)
        idcs = torch.randint(0, 4, (512,), device=device)

        with torch.no_grad():
            out_base = base_dec(coords, latents, conds, idcs_df=idcs)
            out_attn = attn_dec(coords, latents, conds, idcs_df=idcs)

        max_diff = (out_base - out_attn).abs().max().item()
        assert max_diff < 1e-5, f"{ndim}D zero-init mismatch: max_diff={max_diff}"
    print("[PASS] 1. Exact zero-init identity (2D & 3D)")


def test_active_subject_slicing_and_gradient_flow(device):
    torch.manual_seed(7)
    args = make_minimal_args(
        latent_dim=(64, 16, 16),
        out_dim=(2, 2),
        cond_dims=2,
        hidden_size=64,
        num_hidden_layers=3,
        latent_attention={"activate": True, "num_layers": 1, "num_heads": 4, "ffn_mult": 2.0},
        channel_recalibration={"activate": True, "reduction": 4},
    )
    dec = INR_Decoder(args, device).to(device).train()
    latents = torch.nn.Parameter(torch.randn(16, 64, 16, 16, device=device) * 0.05)
    opt = torch.optim.AdamW(list(dec.parameters()) + [latents], lr=1e-2)

    coords = torch.empty(256, 2, device=device).uniform_(-1, 1)
    conds = torch.randn(256, 2, device=device)
    target = torch.randn(256, 4, device=device)

    # Single active subject = 3 out of 16
    idcs_single = torch.full((256,), 3, dtype=torch.long, device=device)
    for step in range(2):
        opt.zero_grad()
        pred = dec(coords, latents, conds, idcs_df=idcs_single)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        # Verify ONLY subject 3 got gradients
        grad_norms = latents.grad.flatten(1).norm(dim=1)
        assert grad_norms[3].item() > 0.0, "Active subject 3 received zero gradient"
        inactive = torch.cat([grad_norms[:3], grad_norms[4:]])
        assert inactive.max().item() == 0.0, f"Inactive subjects leaked gradients: {inactive.max().item()}"

        if step == 1:
            # By step 2 (after zero-init projections stepped once), EVERY attention parameter must have non-zero grad
            for name, p in dec.latent_attention.named_parameters():
                assert p.grad is not None and p.grad.abs().max().item() > 0.0, (
                    f"LatentSpatialAttention param {name} has zero grad at step 2"
                )
            for idx, layer in enumerate(dec.sr_net.net):
                assert layer.channel_recal is not None
                for name, p in layer.channel_recal.named_parameters():
                    assert p.grad is not None and p.grad.abs().max().item() > 0.0, (
                        f"SineLayer[{idx}].channel_recal.{name} has zero grad at step 2"
                    )
        opt.step()

    # Multi-subject batch = {2, 9}
    opt.zero_grad()
    idcs_multi = torch.cat([
        torch.full((128,), 2, dtype=torch.long, device=device),
        torch.full((128,), 9, dtype=torch.long, device=device),
    ])
    pred_multi = dec(coords, latents, conds, idcs_df=idcs_multi)
    torch.nn.functional.mse_loss(pred_multi, target).backward()
    grad_norms = latents.grad.flatten(1).norm(dim=1)
    assert grad_norms[2].item() > 0.0 and grad_norms[9].item() > 0.0
    for s in range(16):
        if s not in (2, 9):
            assert grad_norms[s].item() == 0.0, f"Inactive subject {s} got non-zero grad"

    print("[PASS] 2. Active-subject slicing & full 2-step gradient flow")


def test_tto_spatial_gradient_propagation(device):
    torch.manual_seed(11)
    base_args = make_minimal_args(latent_dim=(64, 16, 16), hidden_size=64, num_hidden_layers=2)
    attn_args = make_minimal_args(
        latent_dim=(64, 16, 16),
        hidden_size=64,
        num_hidden_layers=2,
        latent_attention={"activate": True, "num_layers": 1, "num_heads": 4, "ffn_mult": 2.0},
    )
    base_dec = INR_Decoder(base_args, device).to(device).eval()
    attn_dec = INR_Decoder(attn_args, device).to(device).eval()
    # Give attention output projection small non-zero weights (simulating post-training decoder)
    with torch.no_grad():
        for blk in attn_dec.latent_attention.blocks:
            blk.proj.weight.normal_(0.0, 0.05)

    # Sample coordinates strictly inside top-left quadrant [-1, -0.5] x [-1, -0.5]
    coords_tl = torch.empty(128, 2, device=device).uniform_(-1.0, -0.5)
    conds = torch.randn(128, 2, device=device)

    lat_base = torch.nn.Parameter(torch.randn(1, 64, 16, 16, device=device) * 0.05)
    out_b = base_dec(coords_tl, lat_base, conds)
    out_b.pow(2).mean().backward()
    br_grad_base = lat_base.grad[:, :, 10:, 10:].abs().max().item()
    assert br_grad_base == 0.0, f"Expected 0 bottom-right grad in baseline, got {br_grad_base}"

    lat_attn = torch.nn.Parameter(torch.randn(1, 64, 16, 16, device=device) * 0.05)
    out_a = attn_dec(coords_tl, lat_attn, conds)
    out_a.pow(2).mean().backward()
    br_grad_attn = lat_attn.grad[:, :, 10:, 10:].abs().max().item()
    assert br_grad_attn > 0.0, "Expected non-zero bottom-right grad during TTO with LatentSpatialAttention"
    print(f"[PASS] 3. TTO spatial gradient propagation (baseline BR grad={br_grad_base:.2e}, attn BR grad={br_grad_attn:.2e})")


def test_chunked_inference_equivalence(device):
    torch.manual_seed(99)
    for label, out_dim in [("macustar_faf", (1, 0)), ("multimodal_faf_slo_ga", (2, 2))]:
        args = make_minimal_args(
            latent_dim=(64, 16, 16),
            out_dim=out_dim,
            cond_dims=2,
            hidden_size=64,
            num_hidden_layers=2,
            latent_attention={"activate": True, "num_layers": 1, "num_heads": 4, "ffn_mult": 2.0},
            channel_recalibration={"activate": True, "reduction": 4},
        )
        dec = INR_Decoder(args, device).to(device).eval()
        with torch.no_grad():
            for blk in dec.latent_attention.blocks:
                blk.proj.weight.normal_(0.0, 0.05)
            for layer in dec.sr_net.net:
                layer.channel_recal.fc2.weight.normal_(0.0, 0.05)

        H, W = 64, 64
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        lat = torch.randn(1, 64, 16, 16, device=device)
        cond = torch.randn(1, 2, device=device)

        # Count how many times encode_latent_grids runs during chunked inference
        call_count = {"n": 0}
        orig_encode = dec.encode_latent_grids

        def counted_encode(g):
            call_count["n"] += 1
            return orig_encode(g)

        dec.encode_latent_grids = counted_encode
        with torch.no_grad():
            out_small = dec.inference(coords, lat, cond, [H, W], step_size=512)
            assert call_count["n"] == 1, f"Expected 1 latent-grid attention pass per image, got {call_count['n']}"
            out_full = dec.inference(coords, lat, cond, [H, W], step_size=65536)

        max_diff = (out_small - out_full).abs().max().item()
        assert max_diff < 1e-5, f"{label} chunked inference mismatch: {max_diff}"
    print("[PASS] 4. Chunked inference() exact equivalence & single attention pass per image")


def test_yaml_configs_and_backward_compat(device):
    baseline_cfgs = [
        "configs/config_model.yaml",
        "configs/config_model_macustar_faf.yaml",
        "configs/config_model_faf_slo_ga.yaml",
    ]
    attn_cfgs = [
        "configs/config_model_attention.yaml",
        "configs/config_model_macustar_faf_attention.yaml",
        "configs/config_model_faf_slo_ga_attention.yaml",
    ]

    for rel in baseline_cfgs:
        cfg = load_yaml(rel)
        dec = INR_Decoder(cfg, device).to(device)
        assert isinstance(dec.latent_attention, torch.nn.Identity), f"{rel} should disable latent_attention"
        assert all(layer.channel_recal is None for layer in dec.sr_net.net), f"{rel} should disable channel_recal"
        # Confirm no attention keys in state_dict
        attn_keys = [k for k in dec.state_dict().keys() if "latent_attention" in k or "channel_recal" in k]
        assert len(attn_keys) == 0, f"{rel} leaked attention keys into baseline state_dict: {attn_keys}"

    for rel in attn_cfgs:
        cfg = load_yaml(rel)
        dec = INR_Decoder(cfg, device).to(device)
        assert isinstance(dec.latent_attention, LatentSpatialAttention), f"{rel} should enable latent_attention"
        assert all(isinstance(layer.channel_recal, ChannelRecalibration) for layer in dec.sr_net.net), (
            f"{rel} should enable channel_recal on all trunk SineLayers"
        )
    print("[PASS] 5. Backward compatibility with original configs & new attention configs")


def benchmark_ablation_configs(device):
    base_cfg = load_yaml("configs/config_model_macustar_faf.yaml")
    variants = {
        "Baseline": ({"activate": False}, {"activate": False}),
        "+SRA (LatentAttn 1L 8H)": (
            {"activate": True, "num_layers": 1, "num_heads": 8, "ffn_mult": 2.0, "use_ffn": True},
            {"activate": False},
        ),
        "+CRA (ChannelRecal r=8)": (
            {"activate": False},
            {"activate": True, "reduction": 8},
        ),
        "+SRA+CRA (Full)": (
            {"activate": True, "num_layers": 1, "num_heads": 8, "ffn_mult": 2.0, "use_ffn": True},
            {"activate": True, "reduction": 8},
        ),
    }

    coords = torch.empty(10000, 2, device=device).uniform_(-1, 1)
    latents = torch.randn(10, 256, 32, 32, device=device)
    conds = torch.randn(10000, 2, device=device)
    idcs = torch.zeros(10000, dtype=torch.long, device=device)

    print(f"\n--- 4-Way Ablation Overhead Report (device={device}, latent=[256,32,32], hidden=384x8, N=10000) ---")
    base_params = None
    for name, (la_cfg, cr_cfg) in variants.items():
        cfg = copy.deepcopy(base_cfg)
        cfg["inr_decoder"]["latent_attention"] = la_cfg
        cfg["inr_decoder"]["channel_recalibration"] = cr_cfg
        dec = INR_Decoder(cfg, device).to(device).train()
        n_params = sum(p.numel() for p in dec.parameters())
        if base_params is None:
            base_params = n_params
        delta_pct = 100.0 * (n_params - base_params) / base_params

        # Warmup
        out = dec(coords, latents, conds, idcs_df=idcs)
        out.mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        n_iters = 5 if device.type == "cuda" else 2
        for _ in range(n_iters):
            dec.zero_grad(set_to_none=True)
            out = dec(coords, latents, conds, idcs_df=idcs)
            out.mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = 1000.0 * (time.perf_counter() - t0) / n_iters
        print(f"  {name:<26s} | params = {n_params:>10,d} ({delta_pct:+5.1f}%) | fwd+bwd = {ms:6.1f} ms")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running GAP-INR attention verification suite on {dev}...")
    test_zero_init_identity(dev)
    test_active_subject_slicing_and_gradient_flow(dev)
    test_tto_spatial_gradient_propagation(dev)
    test_chunked_inference_equivalence(dev)
    test_yaml_configs_and_backward_compat(dev)
    benchmark_ablation_configs(dev)
    print("\nALL TESTS PASSED.")
