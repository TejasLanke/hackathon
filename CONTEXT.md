# Project Context Handoff

## Current state

This project is a hackathon MVP for an explainable diabetic retinopathy screening demo. The app is running as a Streamlit application from the project root:

```powershell
streamlit run app.py
```

The app accepts a retinal/fundus image, runs inference through `src.inference.predict`, and displays a prediction, confidence score, and Grad-CAM style visual explanation. It is an educational demo only, not a medical diagnostic tool.

## Architecture

Data flow:

```text
uploaded image -> preprocessing -> EfficientNet-B0/model or demo fallback -> prediction
                                                        |
                                                        v
                                                    Grad-CAM
                                                        |
                                                        v
                                                    Streamlit UI
```

Core files:

- `app.py`: Streamlit UI and user workflow.
- `src/config.py`: central constants for image size, class ids, ImageNet normalization, checkpoint path, and device.
- `src/preprocessing.py`: converts PIL/NumPy images into ImageNet-normalized tensors for EfficientNet-B0.
- `src/model.py`: builds EfficientNet-B0 and loads checkpoint weights.
- `src/inference.py`: stable public prediction API.
- `src/gradcam.py`: Grad-CAM overlay when a model exists, visual fallback in demo mode.
- `src/utils.py`: small shared helpers.
- `training/train.py`: dataset-dependent training entry point.
- `training/evaluate.py`: evaluation metrics and confusion matrix.
- `tests/`: lightweight dataset-independent contract tests.

## Person 4 integration scope

Person 4 work is focused on integration and should stay minimal:

- Keep config centralized in `src/config.py`.
- Keep preprocessing in `src/preprocessing.py`.
- Keep inference behind `src.inference.predict(image)`.
- Keep UI/model/Grad-CAM modules separated.
- Add only lightweight tests that do not require datasets or real model weights.
- Do not download datasets or train models automatically.

## Current inference contract

`src.inference.predict(image)` returns a dictionary with at least:

```python
{
    "class": "no_dr" | "dr",
    "confidence": float,
    "probabilities": {
        "no_dr": float,
        "dr": float,
    },
}
```

It also currently includes UI compatibility metadata:

```python
{
    "label": "No Diabetic Retinopathy" | "Diabetic Retinopathy",
    "class_index": int,
    "is_demo": bool,
}
```

If `models/dr_model.pth` is missing or cannot be loaded, inference fails gracefully into demo mode. Demo-mode values are deterministic and illustrative only.

## Model weights

Expected checkpoint path:

```text
models/dr_model.pth
```

The checkpoint should match the EfficientNet-B0 architecture in `src/model.py` and the class order in `src/config.py`:

```python
CLASS_NAMES = ("no_dr", "dr")
```

Manual placement command:

```powershell
Copy-Item "C:\path\to\trained_weights.pth" "C:\Users\adity\OneDrive\Desktop\Projects\hackathon\models\dr_model.pth"
```

Alternatively, set:

```powershell
$env:DR_MODEL_PATH="C:\full\path\to\your_model.pth"
```

## Dataset layout

Datasets should not be committed. Training expects ImageFolder-style directories:

```text
data/raw/train/no_dr/
data/raw/train/dr/
data/raw/test/no_dr/
data/raw/test/dr/
```

## Run commands

Recommended setup:

```powershell
cd C:\Users\adity\OneDrive\Desktop\Projects\hackathon
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run app:

```powershell
streamlit run app.py
```

Run tests:

```powershell
python -m pytest
```

`pytest.ini` sets `pythonpath = .` so tests can import `src` from the project root.

## Important constraints for future agents

- Do not rewrite teammate-owned modules unless required for integration.
- Do not invent model weights, dataset files, or clinical claims.
- Do not download datasets or train models unless explicitly asked.
- Keep changes hackathon-ready and small.
- Preserve the medical disclaimer and demo-mode messaging.
- If changing class labels, update config, tests, training assumptions, and UI display together.
