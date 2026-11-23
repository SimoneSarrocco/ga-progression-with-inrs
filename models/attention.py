"""
INRAM-inspired attention modules adapted to GAP-INR's latent-grid SIREN architecture
(cf. Yan et al., "INRAM: Implicit Neural Representation with Attention Mechanism",
Sensing and Imaging 26:19, 2025).

1. LatentSpatialAttention (Spatial Recombination Attention analogue):
   Applies Pre-LN Multi-Head Self-Attention with N-D sinusoidal positional encoding
   directly on the regular spatial latent grid z_i (e.g., 32x32 = 1,024 tokens) BEFORE
   coordinate sampling via F.grid_sample. This captures global retinal context (vessels,
   optic disc, multifocal lesions) while remaining 100% invariant to random coordinate
   sampling during training and chunked evaluation during inference.

2. ChannelRecalibration (Channel Recalibration Attention analogue):
   Pointwise dynamic channel gating applied after SineLayer activations, conditioned
   on both the sine features and local latent/condition modulation. Operates strictly
   per-coordinate (O(N), zero train/inference mismatch).

Both modules zero-initialize their residual output projections so that at step 0 the
network is numerically identical to baseline GAP-INR (avoiding early SIREN instability).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_nd_sincos_pos_embed(
    spatial_shape: Tuple[int, ...],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build deterministic N-D (2D or 3D) sinusoidal positional encodings of shape
    (1, prod(spatial_shape), dim).
    """
    ndim = len(spatial_shape)
    if ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D spatial grid, got shape {spatial_shape}")

    # Partition channel budget evenly across spatial axes (pairs of sin/cos per axis)
    base_per_axis = (dim // (2 * ndim)) * 2
    if base_per_axis < 2:
        raise ValueError(f"Channel dim={dim} is too small for {ndim}D sinusoidal PE")

    coords_1d = [
        torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=torch.float32)
        for s in spatial_shape
    ]
    mesh = torch.meshgrid(*coords_1d, indexing="ij")

    parts = []
    used = 0
    for axis_idx, grid_ax in enumerate(mesh):
        # Give any leftover even pairs to the first axes
        rem_axes = ndim - axis_idx
        rem_dim = dim - used
        d_ax = base_per_axis
        if axis_idx == ndim - 1:
            d_ax = (rem_dim // 2) * 2
        half = d_ax // 2
        freq_idx = torch.arange(half, device=device, dtype=torch.float32)
        omega = torch.exp(-math.log(10000.0) * freq_idx / max(half, 1))
        angles = grid_ax.reshape(-1, 1) * omega.unsqueeze(0) * math.pi
        parts.append(torch.sin(angles))
        parts.append(torch.cos(angles))
        used += d_ax

    pe = torch.cat(parts, dim=-1)
    if pe.shape[-1] < dim:
        pad = torch.zeros(pe.shape[0], dim - pe.shape[-1], device=device, dtype=torch.float32)
        pe = torch.cat([pe, pad], dim=-1)
    elif pe.shape[-1] > dim:
        pe = pe[:, :dim]

    return pe.unsqueeze(0).to(dtype=dtype)


class SpatialAttentionBlock(nn.Module):
    """Pre-LN Multi-Head Self-Attention + optional FFN with zero-init residual projections."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_mult: float = 2.0,
        dropout: float = 0.0,
        use_ffn: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Latent channels ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.use_ffn = bool(use_ffn)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        # Zero-init output projection -> exact identity residual at initialization
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        if self.use_ffn:
            hidden = max(int(round(dim * ffn_mult)), dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ffn_fc1 = nn.Linear(dim, hidden, bias=True)
            self.ffn_act = nn.GELU()
            self.ffn_drop = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
            self.ffn_fc2 = nn.Linear(hidden, dim, bias=True)
            nn.init.zeros_(self.ffn_fc2.weight)
            nn.init.zeros_(self.ffn_fc2.bias)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:         (B, N_tokens, C)
            pos_embed: (1, N_tokens, C) optional sinusoidal positional encoding
        """
        B, N, C = x.shape
        h = self.norm1(x)
        if pos_embed is not None:
            h = h + pos_embed

        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        drop_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(attn_out)

        if self.use_ffn:
            h2 = self.ffn_fc2(self.ffn_drop(self.ffn_act(self.ffn_fc1(self.norm2(x)))))
            x = x + h2
        return x


class LatentSpatialAttention(nn.Module):
    """
    Spatial Recombination Attention operating on the per-eye spatial latent grid
    of shape (B, C, H, W) [2D] or (B, C, D, H, W) [3D].
    """

    def __init__(
        self,
        channels: int,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_mult: float = 2.0,
        dropout: float = 0.0,
        use_ffn: bool = True,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.use_pos_embed = bool(use_pos_embed)
        self.blocks = nn.ModuleList([
            SpatialAttentionBlock(
                dim=self.channels,
                num_heads=int(num_heads),
                ffn_mult=float(ffn_mult),
                dropout=float(dropout),
                use_ffn=bool(use_ffn),
            )
            for _ in range(int(num_layers))
        ])
        self._pe_cache: dict[tuple, torch.Tensor] = {}

    def _get_pos_embed(
        self,
        spatial_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.use_pos_embed:
            return None
        key = (spatial_shape, device, dtype)
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = build_nd_sincos_pos_embed(spatial_shape, self.channels, device, dtype)
            # Keep cache tiny (at most a handful of shapes/devices)
            if len(self._pe_cache) > 8:
                self._pe_cache.clear()
            self._pe_cache[key] = pe
        return pe

    def forward(self, latent_grids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_grids: (B, C, *spatial) where spatial is (H, W) or (D, H, W)
        Returns:
            (B, C, *spatial) spatially recombined latent grids.
        """
        orig_shape = latent_grids.shape
        B, C = orig_shape[0], orig_shape[1]
        spatial_shape = tuple(orig_shape[2:])

        # Flatten spatial axes -> (B, N_tokens, C)
        tokens = latent_grids.reshape(B, C, -1).transpose(1, 2).contiguous()
        pe = self._get_pos_embed(spatial_shape, tokens.device, tokens.dtype)

        for idx, blk in enumerate(self.blocks):
            tokens = blk(tokens, pos_embed=pe if idx == 0 else None)

        return tokens.transpose(1, 2).reshape(orig_shape).contiguous()


class ChannelRecalibration(nn.Module):
    """
    INRAM-inspired pointwise Channel Recalibration Attention for SIREN hidden layers.

    Computes a bounded multiplicative channel gate in (0, 2) conditioned on the layer's
    sine features (and optionally the local latent/condition vector):
        gate = 1.0 + tanh(W2(SiLU(W1([h, lat]))))
        out  = h * gate

    W2 and b2 are zero-initialized so gate == 1.0 identically at initialization.
    """

    def __init__(self, out_feat: int, lat_feat: int = 0, reduction: int = 8):
        super().__init__()
        self.out_feat = int(out_feat)
        self.lat_feat = int(lat_feat)
        in_dim = self.out_feat + self.lat_feat
        bottleneck = max(self.out_feat // max(int(reduction), 1), 16)

        self.fc1 = nn.Linear(in_dim, bottleneck, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(bottleneck, self.out_feat, bias=True)

        # Zero-init final projection -> gate = 1 + tanh(0) = 1.0 at step 0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor, lat: torch.Tensor | None = None) -> torch.Tensor:
        if self.lat_feat > 0 and lat is not None:
            ctx = torch.cat([h, lat], dim=-1)
        else:
            ctx = h
        delta = self.fc2(self.act(self.fc1(ctx)))
        gate = 1.0 + torch.tanh(delta)
        return h * gate
