from __future__ import annotations

import math

import torch
from torch import nn


def build_2d_coordinates(
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-major local x/y patch coordinates for a rectangular grid."""
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"Grid must be positive, got {(grid_h, grid_w)}")
    y, x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=dtype),
        torch.arange(grid_w, device=device, dtype=dtype),
        indexing="ij",
    )
    return x.reshape(-1), y.reshape(-1)


def _init_mixed_frequencies(
    depth: int,
    num_heads: int,
    head_dim: int,
    theta: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if head_dim % 4:
        raise ValueError(f"2D Mixed RoPE requires head_dim divisible by 4, got {head_dim}")
    magnitude = 1.0 / (
        theta ** (torch.arange(0, head_dim, 4, dtype=torch.float32) / head_dim)
    )
    per_layer = []
    for _ in range(depth):
        angles = torch.rand(num_heads, 1, generator=generator) * (2 * math.pi)
        freq_x = torch.cat(
            [magnitude * torch.cos(angles), magnitude * torch.cos(angles + math.pi / 2)], dim=-1
        )
        freq_y = torch.cat(
            [magnitude * torch.sin(angles), magnitude * torch.sin(angles + math.pi / 2)], dim=-1
        )
        per_layer.append(torch.stack([freq_x, freq_y], dim=0))
    return torch.stack(per_layer, dim=0)  # [depth, 2, heads, head_dim/2]


class MixedRoPE2D(nn.Module):
    """Learnable per-layer 2D mixed rotary frequencies.

    Coordinates are always local to the current student field of view. Returned
    cos/sin tensors contain spatial tokens only; prefix tokens are deliberately
    excluded and are concatenated unchanged by the attention implementation.
    """

    def __init__(
        self,
        depth: int,
        num_heads: int,
        head_dim: int,
        theta: float = 10.0,
        learnable: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        frequencies = _init_mixed_frequencies(
            depth, num_heads, head_dim, theta, generator=generator
        )
        self.frequencies = nn.Parameter(frequencies, requires_grad=learnable)
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.theta = float(theta)

    def forward(
        self,
        layer_index: int,
        grid_h: int,
        grid_w: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_index < self.depth:
            raise IndexError(layer_index)
        x, y = build_2d_coordinates(grid_h, grid_w, device=device)
        freq = self.frequencies[layer_index].float()
        angles = (
            freq[0].unsqueeze(1) * x.view(1, -1, 1)
            + freq[1].unsqueeze(1) * y.view(1, -1, 1)
        )
        return angles.cos(), angles.sin()  # [heads, spatial_tokens, head_dim/2]


def apply_rotary_spatial(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    prefix_tokens: int,
) -> torch.Tensor:
    """Rotate spatial tokens of [B, heads, tokens, head_dim], preserving prefixes."""
    prefix = tensor[:, :, :prefix_tokens]
    spatial = tensor[:, :, prefix_tokens:]
    if spatial.shape[-2] != cos.shape[-2]:
        raise ValueError(
            f"RoPE spatial length mismatch: tensor={spatial.shape[-2]} coordinates={cos.shape[-2]}"
        )
    original_dtype = spatial.dtype
    pairs = spatial.float().reshape(*spatial.shape[:-1], -1, 2)
    real, imag = pairs.unbind(dim=-1)
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    rotated = torch.stack([real * cos - imag * sin, real * sin + imag * cos], dim=-1)
    rotated = rotated.flatten(-2).to(original_dtype)
    return torch.cat([prefix, rotated], dim=-2)


