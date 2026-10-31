from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class AttentionalPooler(nn.Module):
    """CONCH v1.5 contrastive attentional pooling head."""

    def __init__(self, d_model: int = 768, context_dim: int = 1024, n_head: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model))
        self.heads = n_head
        self.scale = (d_model // n_head) ** -0.5
        self.ln_k = nn.LayerNorm(context_dim)
        self.ln_q = nn.LayerNorm(d_model)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv = nn.Linear(context_dim, d_model * 2, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, context_tokens, _ = x.shape
        query = self.query.view(1, 1, -1).expand(batch, -1, -1)
        query = self.to_q(self.ln_q(query))
        key, value = self.to_kv(self.ln_k(x)).chunk(2, dim=-1)
        head_dim = query.shape[-1] // self.heads
        query = query.reshape(batch, 1, self.heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, context_tokens, self.heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, context_tokens, self.heads, head_dim).transpose(1, 2)
        similarity = (query * self.scale) @ key.transpose(-2, -1)
        similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()
        pooled = similarity.softmax(dim=-1) @ value
        pooled = pooled.transpose(1, 2).reshape(batch, 1, -1)
        return self.to_out(pooled)


class CONCHVisionTower(nn.Module):
    """Exact public CONCH v1.5 vision-only architecture."""

    def __init__(self, image_size: int = 448):
        super().__init__()
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError as exc:
            raise RuntimeError("CONCH construction requires timm") from exc
        self.trunk = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            init_values=1.0,
            num_classes=0,
        )
        self.attn_pool_contrast = AttentionalPooler(768, 1024, 8)
        self.ln_contrast = nn.LayerNorm(768)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.trunk.forward_features(image)
        return self.ln_contrast(self.attn_pool_contrast(tokens)[:, 0])


def resolve_vit_backbone(model: nn.Module) -> nn.Module:
    candidates = [model]
    if hasattr(model, "visual"):
        candidates.insert(0, model.visual)
        if hasattr(model.visual, "trunk"):
            candidates.insert(0, model.visual.trunk)
    if hasattr(model, "trunk"):
        candidates.insert(0, model.trunk)
    for candidate in candidates:
        if hasattr(candidate, "patch_embed") and hasattr(candidate, "blocks"):
            return candidate
    raise TypeError("Could not locate a timm-style ViT backbone (patch_embed + blocks)")


def _to_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def patch_size_of(backbone: nn.Module) -> tuple[int, int]:
    patch_embed = backbone.patch_embed
    value = getattr(patch_embed, "patch_size", None)
    if value is None and hasattr(patch_embed, "proj"):
        value = patch_embed.proj.kernel_size
    if value is None:
        raise AttributeError("Cannot infer patch size")
    return _to_pair(value)


def prefix_token_count(backbone: nn.Module) -> int:
    if hasattr(backbone, "num_prefix_tokens"):
        return int(backbone.num_prefix_tokens)
    count = 0
    cls_token = getattr(backbone, "cls_token", None)
    if cls_token is not None:
        count += int(cls_token.shape[1])
    reg_token = getattr(backbone, "reg_token", None)
    if reg_token is not None:
        count += int(reg_token.shape[1])
    return count


def _infer_pos_grid(backbone: nn.Module) -> tuple[int, int, int]:
    pos = getattr(backbone, "pos_embed", None)
    if pos is None:
        raise ValueError("Teacher has no stored absolute positional embedding")
    prefix = prefix_token_count(backbone)
    spatial = pos.shape[1] - (0 if getattr(backbone, "no_embed_class", False) else prefix)
    side = math.isqrt(spatial)
    if side * side != spatial:
        grid = getattr(backbone.patch_embed, "grid_size", None)
        if grid and int(grid[0]) * int(grid[1]) == spatial:
            return int(grid[0]), int(grid[1]), prefix
        raise ValueError(f"Stored positional embedding spatial count {spatial} is not a square")
    return side, side, prefix


def anchor_teacher_at_512(
    model: nn.Module,
    anchor_size: int = 512,
    interpolate_if_needed: bool = True,
) -> dict[str, Any]:
    backbone = resolve_vit_backbone(model)
    patch_h, patch_w = patch_size_of(backbone)
    if anchor_size % patch_h or anchor_size % patch_w:
        raise ValueError("Anchor size must be divisible by patch size")
    target_grid = (anchor_size // patch_h, anchor_size // patch_w)
    original_h, original_w, prefix = _infer_pos_grid(backbone)
    applied = (original_h, original_w) != target_grid
    if applied and not interpolate_if_needed:
        raise ValueError(
            f"Teacher position grid {(original_h, original_w)} != {target_grid} and interpolation is disabled"
        )
    if applied:
        pos = backbone.pos_embed.detach()
        embedded_prefix = 0 if getattr(backbone, "no_embed_class", False) else prefix
        prefix_pos, spatial_pos = pos[:, :embedded_prefix], pos[:, embedded_prefix:]
        spatial_pos = spatial_pos.reshape(1, original_h, original_w, pos.shape[-1]).permute(0, 3, 1, 2)
        spatial_pos = F.interpolate(
            spatial_pos.float(), size=target_grid, mode="bicubic", align_corners=False
        ).to(pos.dtype)
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).flatten(1, 2)
        backbone.pos_embed = nn.Parameter(torch.cat([prefix_pos, spatial_pos], dim=1))
    patch_embed = backbone.patch_embed
    if hasattr(patch_embed, "img_size"):
        patch_embed.img_size = (anchor_size, anchor_size)
    if hasattr(patch_embed, "grid_size"):
        patch_embed.grid_size = target_grid
    if hasattr(patch_embed, "num_patches"):
        patch_embed.num_patches = target_grid[0] * target_grid[1]
    return {
        "original_pos_grid": [original_h, original_w],
        "target_pos_grid": list(target_grid),
        "pos_interpolation_applied": applied,
    }


def _projection_module(backbone: nn.Module) -> nn.Module | None:
    for name in ("head", "proj", "projection"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module) and not isinstance(module, nn.Identity):
            return module
    return None


def model_preflight(model: nn.Module) -> dict[str, Any]:
    backbone = resolve_vit_backbone(model)
    patch = patch_size_of(backbone)
    blocks = list(backbone.blocks)
    embed_dim = int(
        getattr(backbone, "embed_dim", getattr(backbone, "num_features", 0))
    )
    heads = int(getattr(blocks[0].attn, "num_heads", 0)) if blocks else 0
    pos = getattr(backbone, "pos_embed", None)
    global_pool_module = getattr(model, "attn_pool_contrast", None)
    projection = None if global_pool_module is not None else _projection_module(backbone)
    projected_dim = None
    if hasattr(model, "ln_contrast"):
        projected_dim = int(model.ln_contrast.normalized_shape[0])
    elif projection is not None:
        projected_dim = getattr(projection, "out_features", None)
    return {
        "model_class": type(model).__name__,
        "backbone_class": type(backbone).__name__,
        "patch_size": list(patch),
        "embed_dim": embed_dim,
        "depth": len(blocks),
        "num_heads": heads,
        "stored_positional_embedding_shape": list(pos.shape) if pos is not None else None,
        "number_of_prefix_tokens": prefix_token_count(backbone),
        "trunk_final_norm_module": repr(getattr(backbone, "norm", None)),
        "final_norm_module": repr(getattr(model, "ln_contrast", getattr(backbone, "norm", None))),
        "global_pool_module": repr(global_pool_module),
        "projection_module": repr(projection),
        "public_feature_output_dimension": projected_dim or embed_dim,
    }


def validate_conch_architecture(model: nn.Module, expected_patch: int = 16, expected_depth: int = 24) -> None:
    info = model_preflight(model)
    if info["patch_size"] != [expected_patch, expected_patch]:
        raise RuntimeError(
            f"CONCH checkpoint patch_size={info['patch_size']}, expected {[expected_patch, expected_patch]}"
        )
    if info["depth"] != expected_depth:
        raise RuntimeError(f"CONCH checkpoint depth={info['depth']}, expected {expected_depth}")


def load_conch_teacher(cfg) -> tuple[nn.Module, dict[str, Any]]:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "Loading CONCH requires the optional dependencies: pip install -e '.[conch]'"
        ) from exc
    checkpoint_path = cfg.teacher.checkpoint_path
    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "pytorch_model_vision.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Hugging Face download requires huggingface-hub") from exc
        path = Path(hf_hub_download(cfg.teacher.model_id, filename="pytorch_model_vision.bin"))
    model = CONCHVisionTower(image_size=448)
    state = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected CONCH checkpoint keys: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    anchor_log = anchor_teacher_at_512(
        model,
        anchor_size=int(cfg.teacher.anchor_size),
        interpolate_if_needed=bool(cfg.teacher.ape_interpolation_if_needed),
    )
    validate_conch_architecture(
        model,
        expected_patch=int(cfg.teacher.expected_patch_size),
        expected_depth=int(cfg.teacher.expected_depth),
    )
    model.eval()
    model.requires_grad_(False)
    info = model_preflight(model)
    info.update(anchor_log)
    return model, info

