"""Grad-CAM generation with a visual fallback for model-free demo mode."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from PIL import Image

from src.config import DEVICE
from src.preprocessing import ensure_rgb, preprocess_batch


def _demo_overlay(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    luminance = array.mean(axis=2)
    heat = np.clip((luminance - luminance.min()) / max(np.ptp(luminance), 1.0), 0, 1)
    overlay = array.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], heat * 255)
    overlay[:, :, 1:] *= 1 - (heat[..., None] * 0.35)
    return Image.fromarray(overlay.astype(np.uint8))


def get_gradcam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the final convolutional layer used for EfficientNet-B0 Grad-CAM."""
    features = getattr(model, "features", model)
    for module in reversed(list(features.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("Could not find a convolutional layer for Grad-CAM.")


def _predicted_class_index(model: torch.nn.Module, tensor: torch.Tensor) -> int:
    with torch.inference_mode():
        return int(model(tensor).argmax(dim=1).item())


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

    model.to(DEVICE).eval()
    tensor = preprocess_batch(original).to(DEVICE)
    target_layer = get_gradcam_target_layer(model)
    target_class = _predicted_class_index(model, tensor) if class_index is None else class_index
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    cam_size = (grayscale_cam.shape[1], grayscale_cam.shape[0])
    cam_input = original.resize(cam_size)
    overlay = show_cam_on_image(np.asarray(cam_input, dtype=np.float32) / 255.0, grayscale_cam, use_rgb=True)
    return Image.fromarray(overlay).resize(original.size)
