"""EfficientNet-B0 construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.config import CLASS_NAMES, DEVICE, MODEL_PATH


def build_model(pretrained: bool = False, num_classes: int = len(CLASS_NAMES)) -> nn.Module:
    """Create an EfficientNet-B0 with a binary classification head."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: str | Path | None = None) -> nn.Module | None:
    """Load trained weights when available; otherwise return ``None`` for demo mode."""
    checkpoint_path = Path(model_path) if model_path else MODEL_PATH
    if not checkpoint_path.exists():
        return None

    model = build_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model
