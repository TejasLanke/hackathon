"""Minimal transfer-learning training entry point for ImageFolder datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import BATCH_SIZE, CLASS_NAMES, DEVICE, LEARNING_RATE, MODEL_PATH, NUM_EPOCHS
from src.model import build_model
from src.preprocessing import get_inference_transform
from src.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the diabetic retinopathy classifier.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/raw/test"))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    return parser.parse_args()


def evaluate_loss_and_accuracy(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    if not args.train_dir.exists() or not args.val_dir.exists():
        raise FileNotFoundError("Expected data/raw/train and data/raw/test ImageFolder directories. See README.md.")

    transform = get_inference_transform()
    train_set = datasets.ImageFolder(args.train_dir, transform=transform)
    val_set = datasets.ImageFolder(args.val_dir, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
        val_loss, val_accuracy = evaluate_loss_and_accuracy(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{args.epochs}: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.3f}")

    ensure_directory(MODEL_PATH.parent)
    torch.save({"model_state_dict": model.state_dict()}, MODEL_PATH)
    metadata = {
        "model_name": "efficientnet_b0",
        "class_names": train_set.classes,
        "class_to_idx": train_set.class_to_idx,
        "image_size": 224,
    }
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
