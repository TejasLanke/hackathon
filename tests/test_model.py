from src.model import build_model


def test_model_constructs_binary_head():
    model = build_model(pretrained=False)
    assert model.classifier[1].out_features == 2
