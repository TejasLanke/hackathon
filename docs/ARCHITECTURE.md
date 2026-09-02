# Architecture

```text
                FUNDUS IMAGE
                     |
                     v
             PREPROCESSING
                     |
                     v
             EFFICIENTNET-B0
                /        \
               v          v
          PREDICTION   GRAD-CAM
               |          |
               v          v
          CONFIDENCE   HEATMAP
                \        /
                 v      v
              STREAMLIT
                  |
                  v
           EXPLAINABLE RESULT
```

`src/preprocessing.py` converts uploaded images into ImageNet-normalized tensors. `src/model.py` owns EfficientNet construction and checkpoint loading. `src/inference.py` exposes the stable prediction dictionary and keeps the UI separate from model internals. `src/gradcam.py` generates an overlay for a trained model and a visual demo fallback when no weights exist. `app.py` coordinates the local Streamlit experience. `training/` contains dataset-dependent training and evaluation entry points.
