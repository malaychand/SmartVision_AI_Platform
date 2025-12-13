import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_PATH = "models/detection/best.pt"

@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)

model = load_yolo()

st.title("🎯 Object Detection (YOLOv8)")

conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", width=500)

    results = model.predict(
        source=image,
        conf=conf_threshold,
        save=False
    )

    annotated = results[0].plot()
    st.image(annotated, caption="Detected Objects", width=500)
