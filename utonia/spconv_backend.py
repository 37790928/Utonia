"""
Sparse conv backend for GPU (CUDA + spconv) and Ascend NPU (PyTorch fallback).

**Default:** import ``spconv.pytorch`` first. If that succeeds, use the fast CUDA
sparse conv path. If it fails (typical on Ascend where spconv is unavailable),
fall back to a pure PyTorch submanifold 3x3x3 conv so the same code runs on NPU.

Optional ``UTONIA_USE_SPCONV``:
- unset / empty: try spconv, then fallback (recommended for mixed GPU + NPU setups).
- ``0`` / ``false``: skip importing spconv; always use the PyTorch fallback.
- ``1`` / ``true``: require spconv; re-raise if import fails.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def _env_force_torch_only() -> bool:
    v = os.environ.get("UTONIA_USE_SPCONV", "").strip().lower()
    return v in ("0", "false", "no", "off")


def _env_require_spconv() -> bool:
    v = os.environ.get("UTONIA_USE_SPCONV", "").strip().lower()
    return v in ("1", "true", "yes", "on")


spconv = None
USE_SPCONV = False

if _env_force_torch_only():
    pass
elif _env_require_spconv():
    import spconv.pytorch as spconv  # noqa: F401

    USE_SPCONV = True
else:
    # Default: try GPU spconv first, then Ascend / CPU PyTorch implementation
    try:
        import spconv.pytorch as spconv  # noqa: F401
        print('[DEBUG] spconv loaded')
        USE_SPCONV = True
    except (ImportError, OSError):
        print('[DEBUG] spconv not loaded')
        spconv = None
        USE_SPCONV = False

def is_spconv_module(module: nn.Module) -> bool:
    if USE_SPCONV:
        return spconv.modules.is_spconv_module(module)
    return isinstance(module, SubMConv3dTorch)


# --- Pure PyTorch fallback (SubM 3x3x3, same layout as spconv Conv3d weight) ---


class SparseConvTensorTorch:
    __slots__ = ("features", "indices", "spatial_shape", "batch_size")

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Union[list, tuple, torch.Size],
        batch_size: Union[int, list],
    ):
        self.features = features
        self.indices = indices
        self.spatial_shape = [int(x) for x in spatial_shape]
        self.batch_size = batch_size

    def replace_feature(self, features: torch.Tensor) -> "SparseConvTensorTorch":
        return SparseConvTensorTorch(
            features, self.indices, self.spatial_shape, self.batch_size
        )


def _pack_indices(indices: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Linearize (batch, i0, i1, i2) within the sparse grid for sorting / lookup."""
    D0, D1, D2 = spatial_shape
    b = indices[:, 0].long()
    i0 = indices[:, 1].long()
    i1 = indices[:, 2].long()
    i2 = indices[:, 3].long()
    vol = D0 * D1 * D2
    return b * vol + i0 * (D1 * D2) + i1 * D2 + i2


class SubMConv3dTorch(nn.Module):
    """
    Submanifold sparse 3x3x3 convolution matching spconv.SubMConv3d KRSC weight layout
    (same as checkpoints): [out_ch, k0, k1, k2, in_ch], not nn.Conv3d's [out, in, k, k, k].
    indices [N, 4] as (batch, dim0, dim1, dim2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        indice_key: Optional[str] = None,
    ):
        super().__init__()
        assert kernel_size == 3, "Only 3x3x3 submanifold conv is supported in fallback"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.indice_key = indice_key
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: SparseConvTensorTorch) -> SparseConvTensorTorch:
        feats = x.features
        indices = x.indices
        spatial_shape = tuple(int(s) for s in x.spatial_shape)
        assert len(spatial_shape) == 3

        device = feats.device
        dtype = feats.dtype
        N, C_in = feats.shape
        out_ch = self.out_channels

        if N == 0:
            out = torch.empty(0, out_ch, device=device, dtype=dtype)
            return x.replace_feature(out)

        packed = _pack_indices(indices, spatial_shape)
        sorted_pack, sort_order = torch.sort(packed)

        out_acc = torch.zeros(N, out_ch, device=device, dtype=dtype)
        for k0 in range(3):
            for k1 in range(3):
                for k2 in range(3):
                    d0, d1, d2 = k0 - 1, k1 - 1, k2 - 1
                    neigh = indices.clone()
                    neigh[:, 1] += d0
                    neigh[:, 2] += d1
                    neigh[:, 3] += d2
                    neigh_pack = _pack_indices(neigh, spatial_shape)
                    p = torch.searchsorted(sorted_pack, neigh_pack)
                    # Must not index sorted_pack[p] or sort_order[p] when p == N (OOB on CUDA).
                    valid = p < N
                    ok = torch.zeros(N, dtype=torch.bool, device=device)
                    ok[valid] = sorted_pack[p[valid]] == neigh_pack[valid]
                    src = torch.zeros(N, dtype=torch.long, device=device)
                    src[ok] = sort_order[p[ok]]
                    gathered = torch.zeros(N, C_in, device=device, dtype=dtype)
                    gathered[ok] = feats[src[ok]]
                    w = self.weight[:, k0, k1, k2, :]
                    out_acc += gathered @ w.T

        if self.bias is not None:
            out_acc += self.bias
        return x.replace_feature(out_acc)


if USE_SPCONV:
    SparseConvTensor = spconv.SparseConvTensor
    SubMConv3d = spconv.SubMConv3d
else:
    SparseConvTensor = SparseConvTensorTorch
    SubMConv3d = SubMConv3dTorch
