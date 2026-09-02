"""Grad-CAM generation with a visual fallback for model-free demo mode."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.config import DEVICE
from src.preprocessing import ensure_rgb, preprocess_image


def _demo_overlay(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    luminance = array.mean(axis=2)
    heat = np.clip((luminance - luminance.min()) / max(np.ptp(luminance), 1.0), 0, 1)
    overlay = array.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], heat * 255)
    overlay[:, :, 1:] *= 1 - (heat[..., None] * 0.35)
    return Image.fromarray(overlay.astype(np.uint8))


def generate_gradcam(
    model: torch.nn.Module | None, image: Image.Image | np.ndarray, class_index: int | None = None
) -> Image.Image:
    """Return a Grad-CAM overlay, or a clearly labelled visual fallback without a model."""
    original = ensure_rgb(image)
    if model is None:
        return _demo_overlay(original)

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise RuntimeError("Install grad-cam to generate explanations for a trained model.") from exc

    target_layer = model.features[-1]
    tensor = preprocess_image(original).unsqueeze(0).to(DEVICE)
    targets = [ClassifierOutputTarget(class_index)] if class_index is not None else None
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    resized = original.resize((grayscale_cam.shape[1], grayscale_cam.shape[0]))
    overlay = show_cam_on_image(np.asarray(resized, dtype=np.float32) / 255.0, grayscale_cam, use_rgb=True)
    return Image.fromarray(overlay)
