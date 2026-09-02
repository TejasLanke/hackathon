"""Evaluate a saved model against an ImageFolder test directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import BATCH_SIZE, DEVICE
from src.model import load_model
from src.preprocessing import get_inference_transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DR model.")
    parser.add_argument("--test-dir", type=Path, default=Path("data/raw/test"))
    args = parser.parse_args()
    model = load_model()
    if model is None:
        raise FileNotFoundError("models/dr_model.pth is required before evaluation.")
    dataset = datasets.ImageFolder(args.test_dir, transform=get_inference_transform())
    predictions, targets = [], []
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=BATCH_SIZE):
            predictions.extend(model(images.to(DEVICE)).argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average="binary", zero_division=0)
    print({"accuracy": accuracy_score(targets, predictions), "precision": precision, "recall": recall, "f1": f1})
    print("confusion_matrix:\n", confusion_matrix(targets, predictions))


if __name__ == "__main__":
    main()
