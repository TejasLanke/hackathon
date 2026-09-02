"""Stable prediction interface for trained and demo-mode screening."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.config import CLASS_NAMES, DEVICE, MODEL_PATH
from src.model import load_model
from src.preprocessing import preprocess_batch
from src.utils import display_class_name


@lru_cache(maxsize=1)
def _load_default_model() -> torch.nn.Module | None:
    """Load the configured checkpoint once; missing weights mean demo mode."""
    try:
        return load_model()
    except (FileNotFoundError, RuntimeError, ValueError):
        return None


def _load_class_names(model_path: Path = MODEL_PATH) -> tuple[str, ...]:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return CLASS_NAMES

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CLASS_NAMES

    class_names = metadata.get("class_names")
    if isinstance(class_names, list) and len(class_names) == 2 and all(isinstance(name, str) for name in class_names):
        return tuple(class_names)
    return CLASS_NAMES


def _demo_probabilities(image: Image.Image | np.ndarray) -> np.ndarray:
    """Produce a deterministic visual demo result, never a clinical prediction."""
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    brightness = float(rgb.mean() / 255.0)
    dr_probability = float(np.clip(0.25 + (brightness - 0.5) * 0.2, 0.05, 0.45))
    return np.array([1.0 - dr_probability, dr_probability], dtype=np.float32)


def predict(image: Image.Image | np.ndarray, model: torch.nn.Module | None = None) -> dict[str, Any]:
    """Return the stable prediction contract plus UI metadata."""
    active_model = model if model is not None else _load_default_model()
    class_names = _load_class_names()
    if active_model is None:
        probabilities = _demo_probabilities(image)
        is_demo = True
    else:
        tensor = preprocess_batch(image).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(active_model(tensor), dim=1)[0].cpu().numpy()
        is_demo = False

    class_index = int(np.argmax(probabilities))
    probability_map = {name: float(probabilities[index]) for index, name in enumerate(class_names)}
    class_name = class_names[class_index]
    return {
        "class": class_name,
        "label": display_class_name(class_name),
        "class_index": class_index,
        "confidence": float(probabilities[class_index]),
        "probabilities": probability_map,
        "is_demo": is_demo,
    }
