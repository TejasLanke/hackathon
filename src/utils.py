"""Small shared helpers."""

from __future__ import annotations

from pathlib import Path

from src.config import CLASS_DISPLAY_NAMES


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if necessary and return its path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def display_class_name(class_name: str) -> str:
    """Return a readable label for a stable class id."""
    return CLASS_DISPLAY_NAMES.get(class_name, class_name)
