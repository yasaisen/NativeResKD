from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _as_rgb_array(image: Image.Image | torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    elif torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D image tensor, got {tensor.shape}")
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype.kind == "f" and array.max(initial=0) <= 1.0:
            array = array * 255.0
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {array.shape}")
    return np.clip(array, 0, 255).astype(np.uint8)


def otsu_threshold(values: np.ndarray) -> float:
    values = np.clip(values, 0, 255).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 0.0
    probability = histogram / total
    omega = np.cumsum(probability)
    means = np.cumsum(probability * np.arange(256))
    global_mean = means[-1]
    denominator = omega * (1.0 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > 0
    between[valid] = (global_mean * omega[valid] - means[valid]) ** 2 / denominator[valid]
    return float(np.argmax(between))


def build_tissue_mask(image: Image.Image | torch.Tensor | np.ndarray) -> torch.BoolTensor:
    """HSV-saturation Otsu tissue mask, implemented without an OpenCV dependency."""
    rgb = _as_rgb_array(image).astype(np.float32) / 255.0
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = np.zeros_like(maximum)
    nonzero = maximum > 0
    saturation[nonzero] = (maximum[nonzero] - minimum[nonzero]) / maximum[nonzero]
    saturation_u8 = np.rint(saturation * 255.0).astype(np.uint8)
    threshold = otsu_threshold(saturation_u8)
    # Strict > avoids labeling an all-white, all-zero-saturation image as tissue.
    return torch.from_numpy(saturation_u8 > threshold)


def patch_tissue_ratios(mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected [H,W] tissue mask, got {mask.shape}")
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Tissue mask dimensions must be divisible by patch size")
    return (
        mask.float()
        .reshape(height // patch_size, patch_size, width // patch_size, patch_size)
        .mean(dim=(1, 3))
    )


