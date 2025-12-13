import streamlit as st
import pandas as pd

st.title("📊 Model Performance")

data = {
    "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
    "Accuracy (%)": [83, 88, 85, 92],
    "Inference Time (ms)": [150, 100, 50, 80]
}

df = pd.DataFrame(data)

st.subheader("Classification Performance")
st.dataframe(df, use_container_width=True)

st.bar_chart(df.set_index("Model")[["Accuracy (%)"]])
st.bar_chart(df.set_index("Model")[["Inference Time (ms)"]])
