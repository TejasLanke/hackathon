"""Image conversion and preprocessing utilities."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def ensure_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    """Convert a PIL image or NumPy array into a three-channel RGB PIL image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    return image.convert("RGB")


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_image(image: Image.Image | np.ndarray) -> torch.Tensor:
    """Return a normalized CHW tensor suitable for one-image model inference."""
    return get_inference_transform()(ensure_rgb(image))


def preprocess_batch(image: Image.Image | np.ndarray) -> torch.Tensor:
    """Return a normalized NCHW batch tensor for single-image inference."""
    return preprocess_image(image).unsqueeze(0)
