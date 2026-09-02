import numpy as np
from PIL import Image
from torch import nn

from src.gradcam import generate_gradcam, get_gradcam_target_layer


def test_demo_gradcam_returns_pil_image_at_input_size():
    image = Image.fromarray(np.full((64, 96, 3), 128, dtype=np.uint8))
    overlay = generate_gradcam(None, image)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == image.size


def test_gradcam_target_layer_uses_last_conv_in_features():
    model = nn.Sequential()
    model.features = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(4, 8, kernel_size=3),
    )
    assert get_gradcam_target_layer(model) is model.features[-1]
