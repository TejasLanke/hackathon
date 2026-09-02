"""Central configuration shared by inference, training, and the UI."""

from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "efficientnet_b0"
CHECKPOINT_PATH = Path(os.getenv("DR_MODEL_PATH", PROJECT_ROOT / "models" / "dr_model.pth"))
MODEL_PATH = CHECKPOINT_PATH

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATION_MEAN = IMAGENET_MEAN
NORMALIZATION_STD = IMAGENET_STD

CLASS_NAMES = ("dr", "no_dr")
CLASS_DISPLAY_NAMES = {
    "no_dr": "No Diabetic Retinopathy",
    "dr": "Diabetic Retinopathy",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
