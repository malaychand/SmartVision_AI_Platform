import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/classification/efficientnetb0.h5"
CLASS_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','truck',
    'traffic light','stop sign','bench','bird','cat','dog','horse',
    'cow','elephant','bottle','cup','bowl','pizza','cake','chair',
    'couch','bed','potted plant'
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🧠 Image Classification")

uploaded_file = st.file_uploader(
    "Upload an image (single object)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)

    st.success(f"✅ Prediction: **{CLASS_NAMES[top_idx]}**")
    st.write(f"Confidence: **{preds[top_idx]*100:.2f}%**")

    st.subheader("Top-5 Predictions")
    top5 = np.argsort(preds)[-5:][::-1]
    for i in top5:
        st.write(f"{CLASS_NAMES[i]} : {preds[i]*100:.2f}%")
