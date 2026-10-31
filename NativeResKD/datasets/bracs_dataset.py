from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .kd_dataset import image_to_normalized_tensor, load_metadata

Image.MAX_IMAGE_PIXELS = None  # Trusted official BRACS ROIs can exceed Pillow's warning threshold.


def matched_512_image(image: Image.Image, size: int = 512) -> Image.Image:
    """Centered crop and symmetric white pad. Never resize."""
    image = image.convert("RGB")
    width, height = image.size
    if width > size:
        left = (width - size) // 2
        image = image.crop((left, 0, left + size, height))
    if height > size:
        top = (height - size) // 2
        image = image.crop((0, top, image.width, top + size))
    pad_w, pad_h = size - image.width, size - image.height
    if pad_w or pad_h:
        image = ImageOps.expand(
            image,
            border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
            fill=(255, 255, 255),
        )
    return image


def native_roi_image(image: Image.Image, patch_size: int = 16) -> Image.Image:
    """White-pad bottom/right to the next patch multiple. Never resize."""
    image = image.convert("RGB")
    target_w = math.ceil(image.width / patch_size) * patch_size
    target_h = math.ceil(image.height / patch_size) * patch_size
    return ImageOps.expand(image, border=(0, 0, target_w - image.width, target_h - image.height), fill=(255, 255, 255))


def resample_to_magnification(
    image: Image.Image, source_magnification: float, target_magnification: float
) -> Image.Image:
    """Physical-sampling conversion performed before native-resolution policies."""
    if source_magnification <= 0 or target_magnification <= 0:
        raise ValueError("Magnifications must be positive")
    if source_magnification == target_magnification:
        return image
    scale = target_magnification / source_magnification
    target_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(target_size, resample=Image.Resampling.LANCZOS)


class BRACSDataset(Dataset):
    def __init__(self, metadata: list[dict[str, Any]], cfg, *, split: str, protocol: str):
        if protocol not in {"matched_512", "native_roi"}:
            raise ValueError(protocol)
        if not any("split" in item for item in metadata):
            raise RuntimeError("BRACS metadata has no official split; stop for human confirmation")
        requested_split = split.lower()
        if requested_split in {"valid", "validation"}:
            requested_split = "val"
        self.metadata = [
            item for item in metadata
            if str(item.get("split", "")).lower() == requested_split
        ]
        self.cfg = cfg
        self.split = requested_split
        self.protocol = protocol
        for item in self.metadata:
            if "label" not in item or "image_path" not in item:
                raise KeyError("BRACS metadata requires image_path, label, and split")
            if item.get("magnification") not in (None, "") and float(item["magnification"]) != float(cfg.bracs.source_magnification):
                raise RuntimeError(
                    f"{item.get('sample_id', item['image_path'])} metadata magnification differs from configured source."
                )

    @classmethod
    def from_config(cls, cfg, *, split: str, protocol: str) -> "BRACSDataset":
        if not cfg.bracs.metadata_path:
            raise ValueError("bracs.metadata_path is required")
        return cls(load_metadata(cfg.bracs.metadata_path), cfg, split=split, protocol=protocol)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.metadata[index]
        image = Image.open(item["image_path"]).convert("RGB")
        original_hw = (image.height, image.width)
        source_magnification = float(item.get("magnification") or self.cfg.bracs.source_magnification)
        image = resample_to_magnification(
            image, source_magnification, float(self.cfg.bracs.target_magnification)
        )
        resampled_hw = (image.height, image.width)
        if self.protocol == "matched_512":
            processed = matched_512_image(image)
        else:
            processed = native_roi_image(image, int(self.cfg.teacher.expected_patch_size))
        padded_hw = (processed.height, processed.width)
        return {
            "sample_id": str(item.get("sample_id", index)),
            "image": image_to_normalized_tensor(
                processed, self.cfg.dataset.image_mean, self.cfg.dataset.image_std
            ),
            "label": item["label"],
            "original_hw": torch.tensor(original_hw),
            "resampled_hw": torch.tensor(resampled_hw),
            "padded_hw": torch.tensor(padded_hw),
            "source_magnification": torch.tensor(source_magnification),
            "target_magnification": torch.tensor(float(self.cfg.bracs.target_magnification)),
            "grid_hw": torch.tensor(
                [padded_hw[0] // int(self.cfg.teacher.expected_patch_size), padded_hw[1] // int(self.cfg.teacher.expected_patch_size)]
            ),
        }

