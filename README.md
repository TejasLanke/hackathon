# Explainable AI-based Diabetic Retinopathy Screener

## Problem and solution

Diabetic retinopathy can cause vision loss and needs early assessment. This MVP accepts a retinal/fundus image, screens it with a transfer-learning CNN, and shows a confidence value with a Grad-CAM visual explanation. It is an educational hackathon demo, not a medical device.

## Stack

Python 3.11+, PyTorch, torchvision EfficientNet-B0, Streamlit, scikit-learn, OpenCV, and pytorch-grad-cam.

## Project structure

```text
app.py                 Streamlit demo
src/                   Config, preprocessing, model, inference, Grad-CAM
training/              Training and evaluation scripts
tests/                 Dataset-free contract tests
data/raw/              ImageFolder training and test data (not committed)
models/                Local model checkpoints (not committed)
outputs/               Optional local artifacts
docs/ARCHITECTURE.md   Data flow and module responsibilities
```

## Setup and run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app works without a model checkpoint in clearly marked demo mode. Place trained weights at `models/dr_model.pth` for real inference.

## Dataset and training

Do not commit datasets. Use this ImageFolder layout:

```text
data/raw/train/no_dr/
data/raw/train/dr/
data/raw/test/no_dr/
data/raw/test/dr/
```

Run `python training/train.py` after adding both training and test images. It writes `models/dr_model.pth` and metadata. Run `python training/evaluate.py` to report accuracy, precision, recall, F1, and a confusion matrix.

## Architecture and roles

See `docs/ARCHITECTURE.md` for the data flow and `CONTRIBUTING.md` for branch workflow.

Person 1 owns `training/`, `src/model.py`, and `models/`. Person 2 owns `src/gradcam.py` and its tests. Person 3 owns `app.py`. Person 4 owns preprocessing, inference, configuration, utility modules, dataset layout, and integration tests.

## Limitations and disclaimer

No trained medical model or clinical validation is included. Demo-mode values and visualizations are illustrative only. AI-assisted screening only: this tool is not a medical diagnosis and must not replace assessment by a qualified healthcare professional.
