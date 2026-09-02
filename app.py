"""Streamlit entry point for the explainable DR screening demo."""

from __future__ import annotations

import streamlit as st
from PIL import Image

from src.gradcam import generate_gradcam
from src.inference import predict
from src.model import load_model

st.set_page_config(page_title="DR Screener", page_icon="eye", layout="wide")
st.title("Explainable AI for Diabetic Retinopathy Screening")
st.caption("AI-assisted retinal image screening with visual explanation")
st.warning("AI-assisted screening only. This tool is not a medical diagnosis and should not replace evaluation by a qualified healthcare professional.")

uploaded_file = st.file_uploader("Upload a retinal/fundus image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded fundus image", use_container_width=True)
    if st.button("Analyze Image", type="primary"):
        model = load_model()
        result = predict(image, model=model)
        if result["is_demo"]:
            st.info("Demo mode is active: add models/dr_model.pth to use trained-model predictions.")
        left, right = st.columns(2)
        left.metric("Prediction", result["label"])
        right.metric("Confidence", f"{result['confidence']:.1%}")
        st.subheader("Visual explanation")
        st.image(generate_gradcam(model, image, result["class_index"]), caption="Grad-CAM overlay")
        st.write("Highlighted regions indicate areas that contributed strongly to the model's prediction.")
else:
    st.info("Upload a JPG or PNG fundus image to begin.")
