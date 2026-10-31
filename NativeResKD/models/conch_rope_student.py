from __future__ import annotations

import copy
import types
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .conch_loader import patch_size_of, prefix_token_count, resolve_vit_backbone
from .rope_2d import MixedRoPE2D, apply_rotary_spatial


def _rope_attention_forward(
    attn: nn.Module,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    **_: Any,
) -> torch.Tensor:
    if attn_mask is not None:
        raise RuntimeError("Student attention masks are forbidden; use a real variable-length sequence")
    if is_causal:
        raise RuntimeError("Vision self-attention must not be causal")
    batch, tokens, channels = x.shape
    qkv = attn.qkv(x).reshape(batch, tokens, 3, attn.num_heads, channels // attn.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    rope_state = getattr(attn, "_native_rope_state", None)
    if rope_state is None:
        raise RuntimeError("RoPE attention invoked without coordinates")
    cos, sin, prefixes = rope_state
    q = apply_rotary_spatial(q, cos, sin, prefixes)
    k = apply_rotary_spatial(k, cos, sin, prefixes)
    if getattr(attn, "fused_attn", False) and hasattr(F, "scaled_dot_product_attention"):
        dropout_p = attn.attn_drop.p if attn.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    else:
        scale = getattr(attn, "scale", q.shape[-1] ** -0.5)
        weights = (q * scale) @ k.transpose(-2, -1)
        weights = attn.attn_drop(weights.softmax(dim=-1))
        x = weights @ v
    x = x.transpose(1, 2).reshape(batch, tokens, channels)
    x = attn.proj(x)
    return attn.proj_drop(x)


def _patch_tokens(backbone: nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    patch_h, patch_w = patch_size_of(backbone)
    height, width = image.shape[-2:]
    if height % patch_h or width % patch_w:
        raise ValueError(
            f"Input {(height, width)} must be divisible by patch size {(patch_h, patch_w)}"
        )
    patch_embed = backbone.patch_embed
    if hasattr(patch_embed, "proj"):
        x = patch_embed.proj(image)
        if x.ndim != 4:
            raise RuntimeError(f"Expected convolutional patch output, got {x.shape}")
        x = x.flatten(2).transpose(1, 2)
        norm = getattr(patch_embed, "norm", None)
        if norm is not None:
            x = norm(x)
    else:
        x = patch_embed(image)
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)
    return x, height // patch_h, width // patch_w


def _prefix_tokens(backbone: nn.Module, batch: int) -> torch.Tensor | None:
    values = []
    cls = getattr(backbone, "cls_token", None)
    reg = getattr(backbone, "reg_token", None)
    if cls is not None:
        values.append(cls.expand(batch, -1, -1))
    if reg is not None:
        values.append(reg.expand(batch, -1, -1))
    return torch.cat(values, dim=1) if values else None


def _pool_global(backbone: nn.Module, tokens: torch.Tensor, prefix: int) -> torch.Tensor:
    global_pool = getattr(backbone, "global_pool", "token")
    if global_pool in {"avg", "avgmax"} or prefix == 0:
        value = tokens[:, prefix:].mean(dim=1)
    else:
        value = tokens[:, 0]
    fc_norm = getattr(backbone, "fc_norm", None)
    if isinstance(fc_norm, nn.Module):
        value = fc_norm(value)
    return value


def _global_representations(
    backbone: nn.Module,
    tokens: torch.Tensor,
    trunk_global: torch.Tensor,
    vision_tower: nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if vision_tower is not None and hasattr(vision_tower, "attn_pool_contrast"):
        public_global = vision_tower.attn_pool_contrast(tokens)[:, 0]
        norm = getattr(vision_tower, "ln_contrast", None)
        public_global = norm(public_global) if isinstance(norm, nn.Module) else public_global
        # CONCH v1.5 has no projection after ln_contrast. This normalized 768-d
        # public feature is therefore the required post-final-norm/pre-proj target.
        return public_global, None
    head_drop = getattr(backbone, "head_drop", None)
    projected = head_drop(trunk_global) if isinstance(head_drop, nn.Module) else trunk_global
    for name in ("head", "proj", "projection"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module) and not isinstance(module, nn.Identity):
            return trunk_global, module(projected)
    return trunk_global, None


def forward_vit_features(
    backbone: nn.Module,
    image: torch.Tensor,
    selected_layers: tuple[int, ...],
    *,
    use_ape: bool,
    rope: MixedRoPE2D | None = None,
    vision_tower: nn.Module | None = None,
) -> dict[str, Any]:
    x, grid_h, grid_w = _patch_tokens(backbone, image)
    prefix = prefix_token_count(backbone)
    prefix_values = _prefix_tokens(backbone, image.shape[0])
    pos = getattr(backbone, "pos_embed", None)
    no_embed_class = bool(getattr(backbone, "no_embed_class", False))
    if use_ape:
        if pos is None:
            raise RuntimeError("Teacher forward requested APE but pos_embed is absent")
        if no_embed_class:
            if pos.shape[1] != x.shape[1]:
                raise ValueError(f"APE/token mismatch: {pos.shape[1]} != {x.shape[1]}")
            x = x + pos
            if prefix_values is not None:
                x = torch.cat([prefix_values, x], dim=1)
        else:
            if prefix_values is not None:
                x = torch.cat([prefix_values, x], dim=1)
            if pos.shape[1] != x.shape[1]:
                raise ValueError(f"APE/token mismatch: {pos.shape[1]} != {x.shape[1]}")
            x = x + pos
    elif prefix_values is not None:
        x = torch.cat([prefix_values, x], dim=1)
    pos_drop = getattr(backbone, "pos_drop", None)
    if isinstance(pos_drop, nn.Module):
        x = pos_drop(x)
    patch_drop = getattr(backbone, "patch_drop", None)
    if isinstance(patch_drop, nn.Module):
        x = patch_drop(x)
    norm_pre = getattr(backbone, "norm_pre", None)
    if isinstance(norm_pre, nn.Module):
        x = norm_pre(x)

    hidden_global: dict[int, torch.Tensor] = {}
    hidden_patch: dict[int, torch.Tensor] = {}
    for layer, block in enumerate(backbone.blocks, start=1):
        if rope is not None:
            cos, sin = rope(layer - 1, grid_h, grid_w, device=x.device)
            block.attn._native_rope_state = (cos, sin, prefix)
        x = block(x)
        if layer in selected_layers:
            hidden_global[layer] = _pool_global(backbone, x, prefix)
            hidden_patch[layer] = x[:, prefix:]
    norm = getattr(backbone, "norm", None)
    if isinstance(norm, nn.Module):
        x = norm(x)
    trunk_global = _pool_global(backbone, x, prefix)
    global_preproj, global_projected = _global_representations(
        backbone, x, trunk_global, vision_tower
    )
    return {
        "global_preproj": global_preproj,
        "global_projected": global_projected,
        "trunk_global": trunk_global,
        "hidden_global": hidden_global,
        "hidden_patch": hidden_patch,
        "tokens": x,
        "grid_hw": (grid_h, grid_w),
        "spatial_tokens": grid_h * grid_w,
        "prefix_tokens": prefix,
    }


class TeacherFeatureEncoder(nn.Module):
    def __init__(self, teacher: nn.Module, selected_layers: tuple[int, ...]):
        super().__init__()
        self.vision_tower = teacher
        self.selected_layers = selected_layers
        self.vision_tower.eval().requires_grad_(False)

    @property
    def backbone(self) -> nn.Module:
        return resolve_vit_backbone(self.vision_tower)

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            return forward_vit_features(
                self.backbone,
                image,
                self.selected_layers,
                use_ape=True,
                rope=None,
                vision_tower=self.vision_tower,
            )


class CONCHRoPEStudent(nn.Module):
    def __init__(
        self,
        vision_tower: nn.Module,
        theta: float = 10.0,
        learnable_frequency: bool = True,
        selected_layers: tuple[int, ...] = (6, 12, 18, 24),
        seed: int = 42,
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.selected_layers = selected_layers
        backbone = self.backbone
        blocks = list(backbone.blocks)
        if not blocks:
            raise ValueError("Student backbone has no Transformer blocks")
        heads = int(blocks[0].attn.num_heads)
        embed_dim = int(getattr(backbone, "embed_dim", getattr(backbone, "num_features")))
        self.rope = MixedRoPE2D(
            depth=len(blocks),
            num_heads=heads,
            head_dim=embed_dim // heads,
            theta=theta,
            learnable=learnable_frequency,
            seed=seed,
        )
        # Pure RoPE: unregister the copied APE entirely.
        if hasattr(self.backbone, "pos_embed"):
            self.backbone.pos_embed = None
        for block in blocks:
            if not all(hasattr(block.attn, name) for name in ("qkv", "proj", "num_heads")):
                raise TypeError(f"Unsupported attention module: {type(block.attn).__name__}")
            block.attn.forward = types.MethodType(_rope_attention_forward, block.attn)
        self.requires_grad_(True)

    @property
    def backbone(self) -> nn.Module:
        return resolve_vit_backbone(self.vision_tower)

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        return forward_vit_features(
            self.backbone,
            image,
            self.selected_layers,
            use_ape=False,
            rope=self.rope,
            vision_tower=self.vision_tower,
        )

    def rope_parameters(self):
        return self.rope.parameters()


def surgery_from_teacher(
    teacher: nn.Module,
    *,
    theta: float = 10.0,
    learnable_frequency: bool = True,
    selected_layers: tuple[int, ...] = (6, 12, 18, 24),
    seed: int = 42,
) -> CONCHRoPEStudent:
    student_tower = copy.deepcopy(teacher)
    return CONCHRoPEStudent(
        student_tower,
        theta=theta,
        learnable_frequency=learnable_frequency,
        selected_layers=selected_layers,
        seed=seed,
    )


def verify_surgery_initialization(
    teacher: nn.Module, student: CONCHRoPEStudent
) -> dict[str, list[str]]:
    teacher_state = teacher.state_dict()
    student_state = student.vision_tower.state_dict()
    removed_ape = {key for key in teacher_state if key == "pos_embed" or key.endswith(".pos_embed")}
    missing = sorted(set(teacher_state) - set(student_state) - removed_ape)
    unexpected = sorted(set(student_state) - set(teacher_state))
    mismatched = []
    for key in sorted(set(teacher_state) & set(student_state)):
        if not torch.equal(teacher_state[key], student_state[key]):
            mismatched.append(key)
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Surgery initialization mismatch: missing={missing}, unexpected={unexpected}, values={mismatched}"
        )
    return {"missing_keys": missing, "unexpected_keys": unexpected, "mismatched_keys": mismatched}

