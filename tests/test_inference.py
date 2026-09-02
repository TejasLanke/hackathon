import numpy as np
from PIL import Image

from src.inference import predict


def test_demo_prediction_matches_stable_contract():
    image = Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8))
    result = predict(image, model=None)
    assert {"class", "confidence", "probabilities"}.issubset(result)
    assert result["class"] in {"dr", "no_dr"}
    assert 0 <= result["confidence"] <= 1
    assert set(result["probabilities"]) == {"dr", "no_dr"}
