"""Central configuration shared by inference, training, and the UI."""

from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "efficientnet_b0"
MODEL_PATH = Path(os.getenv("DR_MODEL_PATH", PROJECT_ROOT / "models" / "dr_model.pth"))
IMAGE_SIZE = 224
CLASS_NAMES = ("No Diabetic Retinopathy", "Diabetic Retinopathy")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
