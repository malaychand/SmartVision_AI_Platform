import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PLOT_DIR = "models/result"

MODELS = ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"]

METRICS = {
    "Model":               ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
    "Accuracy (%)":        [83, 88, 85, 92],
    "Precision (%)":       [82, 87, 84, 91],
    "Recall (%)":          [81, 86, 83, 90],
    "F1 Score (%)":        [81, 86, 83, 90],
    "Inference (ms)":      [150, 100, 50, 80],
    "Size (MB)":           [528, 98, 14, 20],
}

TRAINING_FILES = {
    "VGG16":          "vgg16_training.png",
    "ResNet50":       "resnet50_training.png",
    "MobileNetV2":    "mobilenetv2_training_history.png",
    "EfficientNetB0": "efficientnetb0_training.png",
}

CONFUSION_FILES = {
    "VGG16":          "vgg16_confusion_matrix.png",
    "MobileNetV2":    "mobilenetv2_confusion_matrix.png",
}

MODEL_ICONS  = {"VGG16": "🟦", "ResNet50": "🟥", "MobileNetV2": "🟩", "EfficientNetB0": "🟨"}
MODEL_COLORS = {"VGG16": "#3498db", "ResNet50": "#e74c3c", "MobileNetV2": "#2ecc71", "EfficientNetB0": "#f39c12"}

df = pd.DataFrame(METRICS)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Model Performance Dashboard")
st.caption("SmartVision AI · CNN Architecture Comparison · 25-Class COCO Subset")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HERO METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🏆 At a Glance")

c1, c2, c3, c4 = st.columns(4)
card_data = [
    ("Best Accuracy",    "EfficientNetB0", "92%",   "#2ecc71"),
    ("Fastest Model",    "MobileNetV2",    "50 ms",  "#3498db"),
    ("Smallest Size",    "MobileNetV2",    "14 MB",  "#9b59b6"),
    ("Best F1 Score",    "EfficientNetB0", "90%",   "#f39c12"),
]
for col, (label, model, value, color) in zip([c1, c2, c3, c4], card_data):
    with col:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}11);
                border: 2px solid {color}55;
                border-radius: 14px;
                padding: 18px 12px;
                text-align: center;
            ">
                <p style="font-size:12px; color:#888; margin:0; font-weight:600; letter-spacing:1px; text-transform:uppercase;">{label}</p>
                <p style="font-size:26px; font-weight:800; color:{color}; margin:6px 0 2px;">{value}</p>
                <p style="font-size:12px; color:#555; margin:0;">{model}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Full Metrics Table")
st.caption("🟢 Green = best value in each column")

def highlight_best(s):
    if s.name in ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]:
        return ["background-color:#d5f5e3; font-weight:700; color:#1a6b3a"
                if v == s.max() else "" for v in s]
    elif s.name in ["Inference (ms)", "Size (MB)"]:
        return ["background-color:#d5f5e3; font-weight:700; color:#1a6b3a"
                if v == s.min() else "" for v in s]
    return [""] * len(s)

styled = df.style.apply(highlight_best).format({
    "Accuracy (%)":   "{:.0f}%",
    "Precision (%)":  "{:.0f}%",
    "Recall (%)":     "{:.0f}%",
    "F1 Score (%)":   "{:.0f}%",
    "Inference (ms)": "{:.0f} ms",
    "Size (MB)":      "{:.0f} MB",
})
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ACCURACY & INFERENCE CHARTS (side by side)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📈 Accuracy vs Speed")

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(5, 3.8))
    colors = [MODEL_COLORS[m] for m in df["Model"]]
    bars = ax.barh(df["Model"], df["Accuracy (%)"], color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(65, 100)
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_title("Model Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.bar_label(bars, fmt="%.0f%%", padding=4, fontsize=10, fontweight="bold")
    # Star best
    best_y = df["Accuracy (%)"].idxmax()
    ax.get_yticklabels()[best_y].set_fontweight("bold")
    fig.tight_layout()
    st.pyplot(fig)

with right:
    fig2, ax2 = plt.subplots(figsize=(5, 3.8))
    colors2 = [MODEL_COLORS[m] for m in df["Model"]]
    bars2 = ax2.barh(df["Model"], df["Inference (ms)"], color=colors2, edgecolor="white", height=0.55)
    ax2.set_xlabel("Inference Time (ms)", fontsize=10)
    ax2.set_title("Inference Time  ←  Lower is better", fontsize=13, fontweight="bold", pad=10)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(left=False)
    ax2.bar_label(bars2, fmt="%.0f ms", padding=4, fontsize=10, fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — GROUPED PRECISION / RECALL / F1
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🎯 Precision · Recall · F1 Score")

fig3, ax3 = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(df))
w = 0.25

b1 = ax3.bar(x - w, df["Precision (%)"], width=w, label="Precision", color="#3498db", edgecolor="white")
b2 = ax3.bar(x,     df["Recall (%)"],    width=w, label="Recall",    color="#2ecc71", edgecolor="white")
b3 = ax3.bar(x + w, df["F1 Score (%)"], width=w, label="F1 Score",  color="#e74c3c", edgecolor="white")

ax3.set_xticks(x)
ax3.set_xticklabels(df["Model"], fontsize=11)
ax3.set_ylim(70, 100)
ax3.set_ylabel("Score (%)")
ax3.set_title("Precision / Recall / F1 by Model", fontsize=13, fontweight="bold", pad=10)
ax3.legend(framealpha=0.3, fontsize=10)
ax3.spines[["top", "right"]].set_visible(False)
for bars in [b1, b2, b3]:
    ax3.bar_label(bars, fmt="%.0f", padding=2, fontsize=8, fontweight="bold")
fig3.tight_layout()
st.pyplot(fig3)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📉 Training History")
st.caption("Accuracy & loss curves from model training")

available_training = {
    name: os.path.join(PLOT_DIR, fname)
    for name, fname in TRAINING_FILES.items()
    if os.path.exists(os.path.join(PLOT_DIR, fname))
}

if available_training:
    names = list(available_training.keys())
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(names):
                mname = names[i + j]
                color = MODEL_COLORS[mname]
                acc   = df.loc[df["Model"] == mname, "Accuracy (%)"].values[0]
                with col:
                    st.markdown(
                        f"""<div style="border:1.5px solid {color}44; border-radius:12px;
                            padding:12px; background:{color}08; margin-bottom:8px;">
                            <span style="font-size:14px; font-weight:700; color:{color};">
                            {MODEL_ICONS[mname]} {mname}</span>
                            <span style="float:right; font-size:12px; color:#555;">
                            Acc: <b>{acc}%</b></span>
                            </div>""",
                        unsafe_allow_html=True,
                    )
                    st.image(Image.open(available_training[mname]), use_container_width=True)
else:
    st.info(f"📂 Place training PNG files in `{PLOT_DIR}/` (e.g. `vgg16_training.png`)")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔢 Confusion Matrices")
st.caption("Rows = Actual class · Columns = Predicted class · Diagonal = correct predictions")

available_cm = {
    name: os.path.join(PLOT_DIR, fname)
    for name, fname in CONFUSION_FILES.items()
    if os.path.exists(os.path.join(PLOT_DIR, fname))
}

if available_cm:
    names = list(available_cm.keys())
    cols = st.columns(len(names))
    for col, mname in zip(cols, names):
        color = MODEL_COLORS[mname]
        acc   = df.loc[df["Model"] == mname, "Accuracy (%)"].values[0]
        with col:
            st.markdown(
                f"""<div style="border:2px solid {color}66; border-radius:12px;
                    padding:12px 10px 6px; background:{color}08; margin-bottom:6px; text-align:center;">
                    <p style="font-size:15px; font-weight:700; color:{color}; margin:0;">
                    {MODEL_ICONS[mname]} {mname}</p>
                    <p style="font-size:12px; color:#555; margin:4px 0 0;">Accuracy: <b>{acc}%</b></p>
                    </div>""",
                unsafe_allow_html=True,
            )
            st.image(Image.open(available_cm[mname]), use_container_width=True)

else:
    # Fallback: render demo seaborn heatmaps
    st.info(f"📂 Place confusion matrix PNGs in `{PLOT_DIR}/` (e.g. `vgg16_confusion_matrix.png`)")
    CLASS_SHORT = ["person","car","dog","chair","bottle"]
    demo_cms = {
        "VGG16":       np.array([[42,3,2,2,1],[3,44,1,1,1],[2,1,43,2,2],[3,2,2,41,2],[1,1,1,2,45]]),
        "MobileNetV2": np.array([[44,2,1,2,1],[2,46,1,1,0],[1,1,45,2,1],[2,1,1,44,2],[1,0,2,1,46]]),
    }
    cols = st.columns(2)
    for col, (mname, matrix) in zip(cols, demo_cms.items()):
        color = MODEL_COLORS[mname]
        acc   = df.loc[df["Model"] == mname, "Accuracy (%)"].values[0]
        with col:
            st.markdown(
                f"""<div style="border:2px solid {color}66; border-radius:12px;
                    padding:10px; background:{color}08; text-align:center; margin-bottom:6px;">
                    <p style="font-size:15px; font-weight:700; color:{color}; margin:0;">
                    {MODEL_ICONS[mname]} {mname} <span style="font-size:12px;color:#555;">· Acc: {acc}%</span></p>
                    </div>""",
                unsafe_allow_html=True,
            )
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT,
                        linewidths=0.4, linecolor="#e0e0e0",
                        ax=ax_cm, cbar_kws={"shrink": 0.75})
            ax_cm.set_xlabel("Predicted", fontsize=10)
            ax_cm.set_ylabel("Actual", fontsize=10)
            ax_cm.tick_params(axis="x", rotation=30)
            fig_cm.tight_layout()
            st.pyplot(fig_cm)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🧠 Key Takeaways")

t1, t2 = st.columns(2)

with t1:
    st.markdown("""
    | Model | Strength |
    |---|---|
    | 🟨 **EfficientNetB0** | Best accuracy (92%) at tiny size |
    | 🟩 **MobileNetV2** | Fastest (50ms) — ideal for real-time |
    | 🟥 **ResNet50** | Solid all-rounder (88%, 100ms) |
    | 🟦 **VGG16** | Reliable but heavy |
    """)

with t2:
    st.info(
        "💡 **Recommendation:** Use **EfficientNetB0** for maximum accuracy, "
        "or **MobileNetV2** when speed and deployment size matter most "
        "(e.g. edge devices, mobile apps, IoT)."
    )
    st.success("✅ All models exceed the 80% accuracy threshold required for this project.")