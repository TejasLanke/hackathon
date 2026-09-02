import numpy as np
from PIL import Image

from src.config import IMAGE_SIZE
from src.preprocessing import preprocess_batch, preprocess_image


def test_preprocess_image_returns_expected_shape():
    image = Image.fromarray(np.zeros((80, 120, 3), dtype=np.uint8))
    tensor = preprocess_image(image)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_batch_returns_expected_shape():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    tensor = preprocess_batch(image)
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
