import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import time
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# CLASS COLORS — one distinct color per COCO-25 class
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS_BGR = {
    'airplane':      (220,  20,  60),  'bench':        ( 30, 144, 255),
    'bicycle':       (  0, 200,  83),  'bird':         (255, 165,   0),
    'bottle':        (138,  43, 226),  'bowl':         (255,  20, 147),
    'bus':           ( 64, 224, 208),  'cake':         (255, 140,   0),
    'car':           ( 50, 205,  50),  'cat':          (255,   0, 255),
    'chair':         ( 30, 120, 255),  'couch':        (  0, 191, 255),
    'cow':           (  0, 128,   0),  'cup':          (255, 215,   0),
    'dog':           (255,  69,   0),  'elephant':     (128,   0, 128),
    'horse':         (210, 105,  30),  'motorcycle':   (  0, 255, 127),
    'person':        (220,  20,  60),  'pizza':        (255, 165,   0),
    'potted plant':  ( 60, 179, 113),  'stop sign':    (220,  20,  60),
    'traffic light': ( 50, 205,  50),  'train':        (  0, 139, 139),
    'truck':         (  0,   0, 205),
}
DEFAULT_COLOR_BGR = (0, 255, 0)

# Streamlit-friendly hex versions (RGB order for display)
CLASS_COLORS_HEX = {
    'airplane': '#DC143C', 'bench': '#1E90FF', 'bicycle': '#00C853',
    'bird': '#FFA500',     'bottle': '#8B2BE2', 'bowl': '#FF1493',
    'bus': '#40E0D0',      'cake': '#FF8C00',   'car': '#32CD32',
    'cat': '#FF00FF',      'chair': '#1E78FF',  'couch': '#00BFFF',
    'cow': '#008000',      'cup': '#FFD700',    'dog': '#FF4500',
    'elephant': '#800080', 'horse': '#D2691E',  'motorcycle': '#00FF7F',
    'person': '#DC143C',   'pizza': '#FFA500',  'potted plant': '#3CB371',
    'stop sign': '#DC143C','traffic light': '#32CD32', 'train': '#008B8B',
    'truck': '#0000CD',
}
DEFAULT_HEX = '#00FF00'

CLASS_ICONS = {
    'airplane': '✈️', 'bench': '🪑', 'bicycle': '🚲', 'bird': '🐦',
    'bottle': '🍶',   'bowl': '🥣',  'bus': '🚌',   'cake': '🎂',
    'car': '🚗',      'cat': '🐱',   'chair': '🪑', 'couch': '🛋️',
    'cow': '🐮',      'cup': '☕',   'dog': '🐶',   'elephant': '🐘',
    'horse': '🐴',    'motorcycle': '🏍️', 'person': '🧍', 'pizza': '🍕',
    'potted plant': '🪴', 'stop sign': '🛑', 'traffic light': '🚦',
    'train': '🚆',    'truck': '🚛',
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
YOLO_PATH = "models/best_yolo_model.pt"

@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO(YOLO_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPER — clean styled bounding boxes
# ─────────────────────────────────────────────────────────────────────────────
def draw_detections(image_np, boxes, model_names):
    """Draw color-coded bounding boxes with filled label tabs."""
    out = image_np.copy()
    h, w = out.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(w, h) / 1000)
    thickness  = max(2, int(min(w, h) / 300))

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id  = int(box.cls[0])
        conf    = float(box.conf[0])
        label   = model_names[cls_id]
        color   = CLASS_COLORS_BGR.get(label, DEFAULT_COLOR_BGR)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Label background tab
        text    = f" {label}  {conf:.0%}"
        (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
        tab_y1  = max(y1 - th - 10, 0)
        tab_y2  = y1
        cv2.rectangle(out, (x1, tab_y1), (x1 + tw + 4, tab_y2), color, -1)

        # Label text (white)
        cv2.putText(out, text, (x1 + 2, y1 - 5), font,
                    font_scale, (255, 255, 255), thickness)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎯 Object Detection")
st.caption("YOLOv8 · 25-Class COCO Subset · Real-time bounding box detection")

# ── Model status ──────────────────────────────────────────────────────────────
model_exists = os.path.exists(YOLO_PATH) if 'os' in dir() else True
try:
    import os
    model_exists = os.path.exists(YOLO_PATH)
except Exception:
    model_exists = True

col_stat, col_info = st.columns([1, 3])
with col_stat:
    if model_exists:
        st.success("✅ YOLOv8 model loaded")
    else:
        st.error(f"❌ Model not found:\n`{YOLO_PATH}`")

with col_info:
    st.markdown(
        """
        <div style="background:#1a1a2e11; border-radius:10px; padding:10px 14px; font-size:13px; color:#444;">
        🔍 Detects <b>25 object classes</b> from the COCO dataset &nbsp;·&nbsp;
        🎨 Each class gets a unique colour &nbsp;·&nbsp;
        ⚡ Adjust confidence to filter weak detections
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl1, ctrl2 = st.columns([2, 1])
with ctrl1:
    conf_threshold = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.05, max_value=0.95, value=0.40, step=0.05,
        help="Higher = fewer but more certain detections. Lower = catches more objects but may add false positives.",
    )
with ctrl2:
    show_labels = st.toggle("Show Labels", value=True)
    show_stats  = st.toggle("Show Stats Panel", value=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Upload any photo — the model will detect all recognisable objects.",
)

if not uploaded:
    st.info("👆 Upload an image above to run detection.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD + RUN
# ─────────────────────────────────────────────────────────────────────────────
image     = Image.open(uploaded).convert("RGB")
image_np  = np.array(image)

with st.spinner("⏳ Loading YOLOv8…"):
    try:
        model = load_yolo()
    except Exception as e:
        st.error(f"❌ Could not load YOLO model: {e}")
        st.stop()

t0      = time.time()
results = model.predict(
    source=image_np,
    conf=conf_threshold,
    imgsz=640,
    verbose=False,
)[0]
infer_ms = (time.time() - t0) * 1000

boxes = results.boxes
n_det = len(boxes) if boxes is not None else 0

# ─────────────────────────────────────────────────────────────────────────────
# DRAW
# ─────────────────────────────────────────────────────────────────────────────
if n_det > 0 and show_labels:
    annotated = draw_detections(image_np, boxes, model.names)
elif n_det > 0:
    # boxes only, no text
    annotated = image_np.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        color = CLASS_COLORS_BGR.get(label, DEFAULT_COLOR_BGR)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
else:
    annotated = image_np.copy()

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY — image + stats side by side
# ─────────────────────────────────────────────────────────────────────────────
if show_stats and n_det > 0:
    img_col, stat_col = st.columns([3, 1])
else:
    img_col = st.container()
    stat_col = None

with img_col:
    st.image(annotated, channels="RGB", use_container_width=True, caption="Detection Result")

# ─────────────────────────────────────────────────────────────────────────────
# STATS PANEL
# ─────────────────────────────────────────────────────────────────────────────
if show_stats:
    st.divider()
    st.markdown("### 📊 Detection Summary")

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    avg_conf = float(np.mean([float(b.conf[0]) for b in boxes])) if n_det > 0 else 0
    unique_classes = len(set([model.names[int(b.cls[0])] for b in boxes])) if n_det > 0 else 0

    with m1:
        st.markdown(
            f"""<div style="background:#1a73e822; border:2px solid #1a73e855; border-radius:12px;
                padding:14px; text-align:center;">
                <p style="font-size:28px; font-weight:800; color:#1a73e8; margin:0;">{n_det}</p>
                <p style="font-size:12px; color:#555; margin:0;">Objects Found</p></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div style="background:#2ecc7122; border:2px solid #2ecc7155; border-radius:12px;
                padding:14px; text-align:center;">
                <p style="font-size:28px; font-weight:800; color:#2ecc71; margin:0;">{unique_classes}</p>
                <p style="font-size:12px; color:#555; margin:0;">Unique Classes</p></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div style="background:#f39c1222; border:2px solid #f39c1255; border-radius:12px;
                padding:14px; text-align:center;">
                <p style="font-size:28px; font-weight:800; color:#f39c12; margin:0;">{avg_conf:.0%}</p>
                <p style="font-size:12px; color:#555; margin:0;">Avg Confidence</p></div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div style="background:#9b59b622; border:2px solid #9b59b655; border-radius:12px;
                padding:14px; text-align:center;">
                <p style="font-size:28px; font-weight:800; color:#9b59b6; margin:0;">{infer_ms:.0f}ms</p>
                <p style="font-size:12px; color:#555; margin:0;">Inference Time</p></div>""",
            unsafe_allow_html=True,
        )

    if n_det == 0:
        st.warning(
            f"🔍 No objects detected above **{conf_threshold:.0%}** confidence. "
            "Try lowering the threshold slider."
        )
        st.stop()

    st.markdown("")

    # ── Per-detection breakdown ──────────────────────────────────────────────
    st.markdown("#### 🗂️ All Detections")

    # Group by class
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w_box  = x2 - x1
        h_box  = y2 - y1
        detections.append({"label": label, "conf": conf, "w": w_box, "h": h_box})

    detections.sort(key=lambda d: d["conf"], reverse=True)

    # Class count summary badges
    class_counts = Counter(d["label"] for d in detections)
    badge_html   = " &nbsp; ".join(
        f'<span style="background:{CLASS_COLORS_HEX.get(cls, DEFAULT_HEX)}33; '
        f'border:1.5px solid {CLASS_COLORS_HEX.get(cls, DEFAULT_HEX)}88; '
        f'border-radius:20px; padding:3px 10px; font-size:12px; font-weight:600;">'
        f'{CLASS_ICONS.get(cls,"•")} {cls} × {cnt}</span>'
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
    )
    st.markdown(f'<div style="margin-bottom:12px;">{badge_html}</div>', unsafe_allow_html=True)

    # Individual detection rows
    for i, det in enumerate(detections):
        label = det["label"]
        conf  = det["conf"]
        color = CLASS_COLORS_HEX.get(label, DEFAULT_HEX)
        icon  = CLASS_ICONS.get(label, "•")
        bar_w = int(conf * 100)

        st.markdown(
            f"""
            <div style="
                display:flex; align-items:center; gap:12px;
                border-left:4px solid {color}; padding:7px 12px;
                background:{color}0d; border-radius:0 8px 8px 0;
                margin-bottom:5px;
            ">
                <span style="font-size:13px; min-width:20px; color:#888;">#{i+1}</span>
                <span style="font-size:18px;">{icon}</span>
                <span style="font-size:13px; font-weight:600; color:{color}; min-width:120px; text-transform:capitalize;">{label}</span>
                <div style="flex:1; background:#eee; border-radius:6px; height:8px; overflow:hidden;">
                    <div style="width:{bar_w}%; background:{color}; height:100%; border-radius:6px;"></div>
                </div>
                <span style="font-size:13px; font-weight:700; color:{color}; min-width:42px; text-align:right;">{conf:.0%}</span>
                <span style="font-size:11px; color:#999;">{det['w']}×{det['h']}px</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        f"📝 Threshold: {conf_threshold:.0%} · "
        f"Image: {image.size[0]}×{image.size[1]}px · "
        f"Model: YOLOv8 · Inference: {infer_ms:.0f} ms"
    )