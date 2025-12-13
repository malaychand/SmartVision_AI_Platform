import streamlit as st

st.set_page_config(
    page_title="SmartVision",
    layout="wide",
    page_icon="👁️"
)

st.title("👁️ SmartVision")
st.subheader("Unified Image Classification & Object Detection Platform")

st.markdown("""
SmartVision is an end-to-end computer vision system that performs:

- **Multi-class Image Classification** using CNNs  
- **Real-time Object Detection** using YOLOv8  
- **Interactive Web Deployment** using Streamlit  

Navigate using the sidebar to explore different features.
""")

st.info("📌 Use the sidebar to switch between pages.")
