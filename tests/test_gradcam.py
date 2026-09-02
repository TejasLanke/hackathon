import numpy as np
from PIL import Image

from src.gradcam import generate_gradcam


def test_demo_gradcam_returns_pil_image_at_input_size():
    image = Image.fromarray(np.full((64, 96, 3), 128, dtype=np.uint8))
    overlay = generate_gradcam(None, image)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == image.size
