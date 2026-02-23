import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from PIL import Image
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# CLASS NAMES
# Hardcoded from training notebooks (flow_from_directory sorts alphabetically)
# This is the exact order your models learned — no class_indices.json needed.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'airplane', 'bench', 'bicycle', 'bird', 'bottle',
    'bowl', 'bus', 'cake', 'car', 'cat',
    'chair', 'couch', 'cow', 'cup', 'dog',
    'elephant', 'horse', 'motorcycle', 'person', 'pizza',
    'potted plant', 'stop sign', 'traffic light', 'train', 'truck',
]
# Note: flow_from_directory loads classes in alphabetical order.
# The list above is sorted A→Z to match exactly what your models output.

CLASS_ICONS = {
    'airplane': '✈️', 'bench': '🪑', 'bicycle': '🚲', 'bird': '🐦',
    'bottle': '🍶', 'bowl': '🥣', 'bus': '🚌', 'cake': '🎂',
    'car': '🚗', 'cat': '🐱', 'chair': '🪑', 'couch': '🛋️',
    'cow': '🐮', 'cup': '☕', 'dog': '🐶', 'elephant': '🐘',
    'horse': '🐴', 'motorcycle': '🏍️', 'person': '🧍', 'pizza': '🍕',
    'potted plant': '🪴', 'stop sign': '🛑', 'traffic light': '🚦',
    'train': '🚆', 'truck': '🚛',
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGS
# Preprocessing matches exactly what each training notebook used
# ─────────────────────────────────────────────────────────────────────────────
PLOT_DIR = "models/result"

MODEL_CONFIGS = {
    "EfficientNetB0": {
        "path":        os.path.join(PLOT_DIR, "efficientnetb0_best.keras"),
        "preprocess":  eff_preprocess,   # Phase 2.4: preprocess_input from efficientnet
        "rescale":     False,
        "color":       "#f39c12",
        "icon":        "🟨",
        "accuracy":    "92%",
        "speed":       "80 ms",
        "description": "Highest accuracy · Compound scaling",
    },
    "ResNet50": {
        "path":        os.path.join(PLOT_DIR, "resnet50_best.keras"),
        "preprocess":  resnet_preprocess,  # Phase 2.2: preprocess_input from resnet50
        "rescale":     False,
        "color":       "#e74c3c",
        "icon":        "🟥",
        "accuracy":    "88%",
        "speed":       "100 ms",
        "description": "Strong all-rounder · Residual learning",
    },
    "MobileNetV2": {
        "path":        os.path.join(PLOT_DIR, "mobilenetv2_final.h5"),
        "preprocess":  None,               # Phase 2.3: ImageDataGenerator(rescale=1./255)
        "rescale":     True,
        "color":       "#2ecc71",
        "icon":        "🟩",
        "accuracy":    "85%",
        "speed":       "50 ms",
        "description": "Fastest · Lightweight · Edge-ready",
    },
    "VGG16": {
        "path":        os.path.join(PLOT_DIR, "vgg16_best.keras"),
        "preprocess":  vgg_preprocess,     # Phase 2.1: preprocess_input from vgg16
        "rescale":     False,
        "color":       "#3498db",
        "icon":        "🟦",
        "accuracy":    "83%",
        "speed":       "150 ms",
        "description": "Classic CNN · Reliable baseline",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    loaded = {}
    for name, cfg in MODEL_CONFIGS.items():
        if os.path.exists(cfg["path"]):
            try:
                loaded[name] = tf.keras.models.load_model(cfg["path"])
            except Exception as e:
                st.warning(f"⚠️ Failed to load {name}: {e}")
                loaded[name] = None
        else:
            loaded[name] = None
    return loaded


def prepare_image(pil_img, cfg):
    """Resize, apply the correct preprocessing, return (1, 224, 224, 3) array."""
    img = pil_img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    if cfg["rescale"]:
        arr = arr / 255.0            # MobileNetV2 path
    else:
        arr = cfg["preprocess"](arr) # VGG16 / ResNet50 / EfficientNet path
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🖼️ Image Classification")
st.caption("Upload any image — all 4 CNN models classify it simultaneously")

# ── Model status cards ────────────────────────────────────────────────────────
st.markdown("### 🤖 Available Models")
cols = st.columns(4)
for col, (mname, cfg) in zip(cols, MODEL_CONFIGS.items()):
    exists = os.path.exists(cfg["path"])
    status_label = "✅ Ready" if exists else "❌ File missing"
    status_color = "#2ecc71" if exists else "#e74c3c"
    with col:
        st.markdown(
            f"""
            <div style="
                border: 2px solid {cfg['color']}55;
                border-radius: 12px; padding: 12px 10px;
                background: {cfg['color']}0d; text-align: center; height: 148px;
            ">
                <p style="font-size:22px; margin:0;">{cfg['icon']}</p>
                <p style="font-size:13px; font-weight:700; color:{cfg['color']}; margin:4px 0 2px;">{mname}</p>
                <p style="font-size:11px; color:#666; margin:0 0 5px;">{cfg['description']}</p>
                <p style="font-size:11px; color:#444; margin:0;">🎯 {cfg['accuracy']} &nbsp;⚡ {cfg['speed']}</p>
                <p style="font-size:10px; color:{status_color}; margin:5px 0 0; font-weight:600;">{status_label}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📤 Upload an image to classify",
    type=["jpg", "jpeg", "png"],
    help="Best results with a single clear object centred in the frame.",
)

if not uploaded:
    st.info("👆 Upload an image above to see predictions from all 4 models.")
    st.stop()

image = Image.open(uploaded).convert("RGB")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading models (cached after first run)…"):
    models = load_all_models()

available = {k: v for k, v in models.items() if v is not None}

if not available:
    st.error(
        "❌ No model files found in `model/result/`. "
        "Make sure `.keras` / `.h5` files are present."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
all_results = {}
bar = st.progress(0, text="Running inference…")

for i, (mname, model) in enumerate(available.items()):
    cfg = MODEL_CONFIGS[mname]
    arr = prepare_image(image, cfg)

    t0    = time.time()
    preds = model.predict(arr, verbose=0)[0]
    ms    = (time.time() - t0) * 1000

    top5_idx = np.argsort(preds)[::-1][:5]
    top5     = [(CLASS_NAMES[j], float(preds[j])) for j in top5_idx]

    all_results[mname] = {"top5": top5, "top1": top5[0], "ms": ms, "cfg": cfg}
    bar.progress((i + 1) / len(available), text=f"✅ {mname} complete")

bar.empty()

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT: image | consensus
# ─────────────────────────────────────────────────────────────────────────────
img_col, sum_col = st.columns([1, 1])

with img_col:
    st.image(image, caption="📷 Input Image", use_container_width=True)
    st.caption(f"Size: {image.size[0]}×{image.size[1]} px · RGB")

with sum_col:
    st.markdown("### 🧠 Model Consensus")

    votes     = [r["top1"][0] for r in all_results.values()]
    consensus = max(set(votes), key=votes.count)
    n_agree   = votes.count(consensus)
    icon      = CLASS_ICONS.get(consensus, "🔍")
    c_color   = "#2ecc71" if n_agree >= 3 else "#f39c12" if n_agree == 2 else "#e74c3c"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {c_color}22, {c_color}11);
            border: 2px solid {c_color}66; border-radius: 14px;
            padding: 22px; text-align: center; margin-bottom: 16px;
        ">
            <p style="font-size:52px; margin:0;">{icon}</p>
            <p style="font-size:22px; font-weight:800; color:{c_color};
               margin:8px 0 4px; text-transform:capitalize;">{consensus}</p>
            <p style="font-size:13px; color:#555; margin:0;">{n_agree} / {len(all_results)} models agree</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Top prediction per model:**")
    for mname, res in all_results.items():
        cfg    = res["cfg"]
        label  = res["top1"][0]
        conf   = res["top1"][1]
        match  = "✅" if label == consensus else "🔶"
        st.markdown(
            f"""
            <div style="
                display:flex; justify-content:space-between; align-items:center;
                border-left: 4px solid {cfg['color']}; padding: 6px 10px;
                margin-bottom: 6px; background: {cfg['color']}0d;
                border-radius: 0 8px 8px 0;
            ">
                <span style="font-size:13px; font-weight:600; color:{cfg['color']};">{cfg['icon']} {mname}</span>
                <span style="font-size:13px;">{match} {label}</span>
                <span style="font-size:12px; color:#888;">{conf:.0%} · {res['ms']:.0f}ms</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# DETAILED TOP-5 CARDS — 2-column grid
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Detailed Predictions — Top 5 Per Model")

names = list(all_results.keys())
for i in range(0, len(names), 2):
    row = st.columns(2)
    for j, col in enumerate(row):
        if i + j >= len(names):
            break
        mname = names[i + j]
        res   = all_results[mname]
        cfg   = res["cfg"]

        with col:
            # Header
            st.markdown(
                f"""
                <div style="border:2px solid {cfg['color']}66; border-radius:12px 12px 0 0;
                    padding:10px 14px; background:{cfg['color']}15;">
                    <span style="font-size:15px; font-weight:700; color:{cfg['color']};">{cfg['icon']} {mname}</span>
                    <span style="float:right; font-size:12px; color:#666;">⚡ {res['ms']:.0f} ms</span><br>
                    <span style="font-size:11px; color:#666;">{cfg['description']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top-5 bars
            for rank, (label, conf) in enumerate(res["top5"]):
                li    = CLASS_ICONS.get(label, "•")
                bar_c = cfg["color"] if rank == 0 else "#bbb"
                bg    = f"{cfg['color']}18" if rank == 0 else "transparent"
                fw    = "700" if rank == 0 else "400"
                medal = "🥇" if rank == 0 else f"#{rank+1}"

                st.markdown(
                    f"""
                    <div style="padding:5px 14px; background:{bg};">
                        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                            <span style="font-size:13px; font-weight:{fw};">{medal} {li} {label}</span>
                            <span style="font-size:12px; font-weight:{fw}; color:{bar_c};">{conf:.1%}</span>
                        </div>
                        <div style="background:#eee; border-radius:6px; height:7px; overflow:hidden;">
                            <div style="width:{max(int(conf*100),2)}%; background:{bar_c};
                                height:100%; border-radius:6px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Footer border
            st.markdown(
                f'<div style="border:2px solid {cfg["color"]}44; border-top:none; '
                f'border-radius:0 0 12px 12px; height:8px;"></div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LIVE INFERENCE TIME — vertical bars
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### ⚡ Live Inference Time")
max_ms = max(r["ms"] for r in all_results.values()) or 1

speed_cols = st.columns(len(all_results))
for col, (mname, res) in zip(speed_cols, all_results.items()):
    cfg = res["cfg"]
    pct = int((res["ms"] / max_ms) * 100)
    with col:
        st.markdown(
            f"""
            <div style="text-align:center; padding:10px;">
                <p style="font-size:12px; font-weight:600; color:{cfg['color']}; margin:0 0 6px;">{cfg['icon']} {mname}</p>
                <div style="background:#eee; border-radius:8px; height:80px; position:relative; overflow:hidden;">
                    <div style="position:absolute; bottom:0; width:100%; height:{pct}%;
                        background:{cfg['color']}; border-radius:8px;"></div>
                </div>
                <p style="font-size:13px; font-weight:700; margin:6px 0 0; color:{cfg['color']};">{res['ms']:.0f} ms</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("📝 Inference times measured live on this machine. GPU will be significantly faster.")