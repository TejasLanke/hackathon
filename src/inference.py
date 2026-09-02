"""Stable prediction interface for trained and demo-mode screening."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from src.config import CLASS_NAMES, DEVICE
from src.model import load_model
from src.preprocessing import preprocess_image


def _demo_probabilities(image: Image.Image | np.ndarray) -> np.ndarray:
    """Produce a deterministic visual demo result, never a clinical prediction."""
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.float32)
    brightness = float(rgb.mean() / 255.0)
    dr_probability = float(np.clip(0.25 + (brightness - 0.5) * 0.2, 0.05, 0.45))
    return np.array([1.0 - dr_probability, dr_probability])


def predict(image: Image.Image | np.ndarray, model: torch.nn.Module | None = None) -> dict[str, Any]:
    """Return label, class index, confidence, probabilities, and demo-mode status."""
    active_model = model if model is not None else load_model()
    if active_model is None:
        probabilities = _demo_probabilities(image)
        is_demo = True
    else:
        tensor = preprocess_image(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(active_model(tensor), dim=1)[0].cpu().numpy()
        is_demo = False

    class_index = int(np.argmax(probabilities))
    probability_map = {name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)}
    return {
        "label": CLASS_NAMES[class_index],
        "class_index": class_index,
        "confidence": float(probabilities[class_index]),
        "probabilities": probability_map,
        "is_demo": is_demo,
    }
