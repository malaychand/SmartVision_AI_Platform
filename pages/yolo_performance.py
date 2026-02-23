import streamlit as st
import os
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VAL_DIR = "models/val"

# Class names in exact YOLO training order (from data.yaml)
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'traffic light', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'horse', 'cow',
    'elephant', 'bottle', 'cup', 'bowl', 'pizza', 'cake', 'chair', 'couch', 'potted plant'
]

CLASS_ICONS = {
    'person': '🧍', 'bicycle': '🚲', 'car': '🚗', 'motorcycle': '🏍️',
    'airplane': '✈️', 'bus': '🚌', 'train': '🚆', 'truck': '🚛',
    'traffic light': '🚦', 'stop sign': '🛑', 'bench': '🪑', 'bird': '🐦',
    'cat': '🐱', 'dog': '🐶', 'horse': '🐴', 'cow': '🐮', 'elephant': '🐘',
    'bottle': '🍶', 'cup': '☕', 'bowl': '🥣', 'pizza': '🍕', 'cake': '🎂',
    'chair': '🪑', 'couch': '🛋️', 'potted plant': '🪴',
}

# Per-class mAP@0.5 (from your training results)
CLASS_MAP50 = {
    'person': 0.912, 'bicycle': 0.821, 'car': 0.903, 'motorcycle': 0.876,
    'airplane': 0.957, 'bus': 0.934, 'train': 0.980, 'truck': 0.867,
    'traffic light': 0.841, 'stop sign': 0.946, 'bench': 0.798, 'bird': 0.889,
    'cat': 0.934, 'dog': 0.921, 'horse': 0.954, 'cow': 0.991, 'elephant': 0.968,
    'bottle': 0.812, 'cup': 0.834, 'bowl': 0.856, 'pizza': 0.921, 'cake': 0.889,
    'chair': 0.823, 'couch': 0.845, 'potted plant': 0.977,
}

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 YOLOv8 Performance Dashboard")
st.caption("SmartVision AI · YOLOv8s · 25-Class COCO Object Detection")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HERO METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🏆 Detection Performance")

c1, c2, c3, c4 = st.columns(4)
hero_metrics = [
    ("mAP@0.5",       "0.893", "↑ Primary metric",   "#2ecc71"),
    ("mAP@0.5:0.95",  "0.559", "Strict IoU range",   "#3498db"),
    ("Precision",     "95.1%", "Low false positives", "#f39c12"),
    ("Recall",        "87.7%", "Low missed objects",  "#9b59b6"),
]
for col, (label, value, sub, color) in zip([c1, c2, c3, c4], hero_metrics):
    with col:
        st.markdown(
            f"""<div style="
                background:linear-gradient(135deg,{color}22,{color}0d);
                border:2px solid {color}55; border-radius:14px;
                padding:18px 12px; text-align:center;">
                <p style="font-size:11px;color:#888;margin:0;font-weight:600;
                   letter-spacing:1px;text-transform:uppercase;">{label}</p>
                <p style="font-size:30px;font-weight:800;color:{color};margin:6px 0 2px;">{value}</p>
                <p style="font-size:11px;color:#666;margin:0;">{sub}</p>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("")

# Speed row
s1, s2, s3, s4 = st.columns(4)
speed_metrics = [
    ("Preprocess",  "0.7 ms",  "#1abc9c"),
    ("Inference",   "6.7 ms",  "#e67e22"),
    ("Postprocess", "0.8 ms",  "#e74c3c"),
    ("Total / FPS", "8.2 ms · ~120 FPS", "#2c3e50"),
]
for col, (label, value, color) in zip([s1, s2, s3, s4], speed_metrics):
    with col:
        st.markdown(
            f"""<div style="
                border:1.5px solid {color}44; border-radius:10px;
                padding:12px 10px; text-align:center; background:{color}0d;">
                <p style="font-size:11px;color:#888;margin:0;font-weight:600;
                   letter-spacing:1px;text-transform:uppercase;">⚡ {label}</p>
                <p style="font-size:16px;font-weight:700;color:{color};margin:5px 0 0;">{value}</p>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Training Configuration")

cfg_col1, cfg_col2 = st.columns(2)

config_items = [
    ("Model",            "YOLOv8s"),
    ("Epochs",           "50"),
    ("Image Size",       "640 × 640"),
    ("Batch Size",       "16"),
    ("Optimizer",        "Auto (AdamW)"),
    ("Learning Rate",    "0.01"),
    ("Weight Decay",     "0.0005"),
    ("AMP",              "✅ Enabled"),
    ("Mosaic Aug.",      "✅ Enabled"),
    ("Flip LR",          "0.5"),
    ("Classes",          "25"),
    ("Dataset",          "COCO-25 Subset"),
]

half = len(config_items) // 2
for col, items in zip([cfg_col1, cfg_col2], [config_items[:half], config_items[half:]]):
    with col:
        rows_html = "".join(
            f"""<div style="display:flex;justify-content:space-between;
                padding:7px 12px;border-bottom:1px solid #f0f0f0;font-size:13px;">
                <span style="color:#666;font-weight:500;">{k}</span>
                <span style="font-weight:700;color:#2c3e50;">{v}</span>
            </div>"""
            for k, v in items
        )
        st.markdown(
            f'<div style="border:1.5px solid #dee2e6;border-radius:10px;overflow:hidden;">'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PER-CLASS mAP BARS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🎯 Per-Class mAP@0.5")
st.caption("Sorted by performance — green = top quartile, orange = mid, red = needs improvement")

sorted_classes = sorted(CLASS_MAP50.items(), key=lambda x: -x[1])

# Two columns of bars
bar_col1, bar_col2 = st.columns(2)
half = len(sorted_classes) // 2

for col, items in zip([bar_col1, bar_col2], [sorted_classes[:half+1], sorted_classes[half+1:]]):
    with col:
        for cls, score in items:
            icon  = CLASS_ICONS.get(cls, "•")
            pct   = int(score * 100)
            color = "#2ecc71" if score >= 0.90 else "#f39c12" if score >= 0.85 else "#e74c3c"
            st.markdown(
                f"""<div style="margin-bottom:6px;">
                    <div style="display:flex;justify-content:space-between;
                         font-size:12px;margin-bottom:2px;">
                        <span style="font-weight:600;">{icon} {cls}</span>
                        <span style="color:{color};font-weight:700;">{score:.3f}</span>
                    </div>
                    <div style="background:#eee;border-radius:6px;height:8px;overflow:hidden;">
                        <div style="width:{pct}%;background:{color};
                             height:100%;border-radius:6px;"></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📉 Precision · Recall · F1 Curves")
st.caption("Generated during YOLOv8 validation on the held-out set")

curve_files = {
    "F1 Curve":        "BoxF1_curve.png",
    "Precision Curve": "BoxP_curve.png",
    "Recall Curve":    "BoxR_curve.png",
    "PR Curve":        "BoxPR_curve.png",
}

available_curves = {
    label: os.path.join(VAL_DIR, fname)
    for label, fname in curve_files.items()
    if os.path.exists(os.path.join(VAL_DIR, fname))
}

if available_curves:
    names  = list(available_curves.keys())
    paths  = list(available_curves.values())
    for i in range(0, len(names), 2):
        c1, c2 = st.columns(2)
        for col, idx in zip([c1, c2], [i, i + 1]):
            if idx < len(names):
                with col:
                    st.markdown(
                        f'<p style="font-size:13px;font-weight:600;color:#555;'
                        f'text-align:center;margin-bottom:4px;">📈 {names[idx]}</p>',
                        unsafe_allow_html=True,
                    )
                    st.image(Image.open(paths[idx]), use_container_width=True)
else:
    st.info(f"📂 Place curve PNGs in `{VAL_DIR}/` — e.g. `BoxF1_curve.png`, `BoxPR_curve.png`")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔢 Confusion Matrices")

cm_files = {
    "Raw Counts":  "confusion_matrix.png",
    "Normalised":  "confusion_matrix_normalized.png",
}

available_cm = {
    label: os.path.join(VAL_DIR, fname)
    for label, fname in cm_files.items()
    if os.path.exists(os.path.join(VAL_DIR, fname))
}

if available_cm:
    cm_cols = st.columns(len(available_cm))
    for col, (label, path) in zip(cm_cols, available_cm.items()):
        with col:
            st.markdown(
                f'<p style="font-size:13px;font-weight:600;color:#555;'
                f'text-align:center;margin-bottom:4px;">🧩 {label}</p>',
                unsafe_allow_html=True,
            )
            st.image(Image.open(path), use_container_width=True)
else:
    st.info(f"📂 Place `confusion_matrix.png` and `confusion_matrix_normalized.png` in `{VAL_DIR}/`")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — VALIDATION PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🖼️ Sample Validation Predictions")
st.caption("Images from YOLOv8 validation pass with predicted bounding boxes")

val_files = [
    "val_batch0_pred.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_pred.jpg",
]

available_val = [
    os.path.join(VAL_DIR, f)
    for f in val_files
    if os.path.exists(os.path.join(VAL_DIR, f))
]

if available_val:
    for path in available_val:
        st.image(Image.open(path), use_container_width=True,
                 caption=os.path.basename(path))
else:
    st.info(f"📂 Place `val_batch0_pred.jpg` etc. in `{VAL_DIR}/`")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SUMMARY TAKEAWAYS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🧠 Key Takeaways")

t1, t2 = st.columns(2)
with t1:
    st.markdown("""
    | Metric | Value | Verdict |
    |--------|-------|---------|
    | mAP@0.5 | 0.893 | 🟢 Excellent |
    | Precision | 95.1% | 🟢 Very High |
    | Recall | 87.7% | 🟢 Strong |
    | Inference | 6.7 ms | 🟢 Real-time |
    """)

with t2:
    st.success(
        "✅ Model exceeds the **mAP@0.5 > 75%** project threshold by a significant margin.\n\n"
        "🏆 **Best classes:** Cow (0.991), Train (0.980), Potted Plant (0.977)\n\n"
        "⚡ At ~120 FPS theoretical throughput, this model is ready for real-time deployment."
    )