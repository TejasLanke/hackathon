import numpy as np
from PIL import Image

from src.config import CLASS_NAMES
from src.inference import predict


def test_demo_prediction_matches_stable_contract():
    image = Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8))
    result = predict(image, model=None)
    assert set(result) == {"label", "class_index", "confidence", "probabilities", "is_demo"}
    assert result["label"] in CLASS_NAMES
    assert 0 <= result["confidence"] <= 1
    assert set(result["probabilities"]) == set(CLASS_NAMES)
