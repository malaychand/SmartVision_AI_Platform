import streamlit as st
import cv2
import time
import numpy as np
import os
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# CLASS COLORS — one distinct BGR color per COCO-25 class
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS_BGR = {
    'person': (60, 20, 220),        'bicycle': (0, 200, 83),
    'car': (50, 205, 50),           'motorcycle': (0, 255, 127),
    'airplane': (220, 20, 60),      'bus': (64, 224, 208),
    'train': (0, 139, 139),         'truck': (0, 0, 205),
    'traffic light': (50, 205, 50), 'stop sign': (0, 0, 220),
    'bench': (30, 144, 255),        'bird': (255, 165, 0),
    'cat': (255, 0, 255),           'dog': (255, 69, 0),
    'horse': (210, 105, 30),        'cow': (0, 128, 0),
    'elephant': (128, 0, 128),      'bottle': (138, 43, 226),
    'cup': (255, 215, 0),           'bowl': (255, 20, 147),
    'pizza': (255, 140, 0),         'cake': (255, 192, 0),
    'chair': (30, 120, 255),        'couch': (0, 191, 255),
    'potted plant': (60, 179, 113),
}
DEFAULT_COLOR = (0, 255, 0)

YOLO_PATH = "models/best_yolo_model.pt"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO(YOLO_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_frame(frame, results, show_conf=True):
    """Draw color-coded bounding boxes with filled label tabs."""
    h, w   = frame.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.4, min(w, h) / 1200)
    thick  = max(1, int(min(w, h) / 400))

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = results.names[cls_id]
        color  = CLASS_COLORS_BGR.get(label, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick + 1)

        text = f" {label} {conf:.0%}" if show_conf else f" {label}"
        (tw, th), _ = cv2.getTextSize(text, font, fscale, thick)
        tab_y = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, tab_y), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), font,
                    fscale, (255, 255, 255), thick)
    return frame


def draw_fps_overlay(frame, fps, n_det):
    """Transparent FPS + object count overlay in top-left."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (210, 62), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}",     (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects: {n_det}",   (16, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    return frame

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📷 Live Webcam Detection")
st.caption("YOLOv8 · Real-time object detection · Color-coded per class")

st.info(
    "🖥️ **Local execution only.** Webcam capture requires direct hardware access. "
    "On cloud deployments (Hugging Face Spaces), use **🎯 Object Detection** with image upload instead."
)

# Model status
model_exists = os.path.exists(YOLO_PATH)
if model_exists:
    st.success(f"✅ YOLOv8 model ready: `{YOLO_PATH}`")
else:
    st.error(f"❌ Model not found at `{YOLO_PATH}`. Copy `best.pt` from your training output.")
    st.stop()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns(3)

with ctrl1:
    conf_threshold = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.10, max_value=0.95, value=0.45, step=0.05,
        help="Lower = more detections, Higher = more certain detections only",
    )
with ctrl2:
    show_conf = st.toggle("Show Confidence %", value=True)
    show_fps  = st.toggle("Show FPS Overlay",  value=True)
with ctrl3:
    camera_idx = st.selectbox(
        "📷 Camera Index", [0, 1, 2], index=0,
        help="0 = default webcam. Try 1 or 2 for USB cameras.",
    )
    target_fps = st.selectbox("🎯 Target FPS Cap", [15, 30, 60], index=1)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# START / STOP BUTTONS
# ─────────────────────────────────────────────────────────────────────────────
b1, b2, _ = st.columns([1, 1, 3])
with b1:
    start = st.button("▶️ Start Webcam", type="primary",    use_container_width=True)
with b2:
    stop  = st.button("⏹️ Stop",         type="secondary",  use_container_width=True)

frame_ph = st.empty()
stats_ph = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# WEBCAM LOOP
# ─────────────────────────────────────────────────────────────────────────────
if start:
    with st.spinner("⏳ Loading YOLOv8…"):
        try:
            model = load_yolo()
        except Exception as e:
            st.error(f"❌ Model load failed: {e}")
            st.stop()

    cap = cv2.VideoCapture(int(camera_idx))

    if not cap.isOpened():
        st.error(
            "❌ Could not open webcam. "
            "Ensure your camera is connected and not in use by another app."
        )
        st.stop()

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_gap = 1.0 / target_fps
    prev_time = time.time()
    frame_num = 0

    st.success("🟢 Webcam running — press **⏹️ Stop** to end.")

    while cap.isOpened():
        if stop:
            break

        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Lost webcam feed. Retrying…")
            time.sleep(0.1)
            continue

        # ── Inference ────────────────────────────────────────────────────────
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            imgsz=640,
            verbose=False,
        )[0]

        n_det = len(results.boxes) if results.boxes is not None else 0

        # ── Draw ─────────────────────────────────────────────────────────────
        annotated = draw_frame(frame.copy(), results, show_conf=show_conf)

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if show_fps:
            annotated = draw_fps_overlay(annotated, fps, n_det)

        # ── Render ───────────────────────────────────────────────────────────
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_ph.image(frame_rgb, channels="RGB", use_container_width=True)

        with stats_ph.container():
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("⚡ FPS",       f"{fps:.1f}")
            s2.metric("🔍 Objects",   f"{n_det}")
            s3.metric("🎚️ Threshold", f"{conf_threshold:.0%}")
            s4.metric("🎞️ Frame",     f"#{frame_num}")

        frame_num += 1

        # FPS cap
        elapsed = time.time() - loop_start
        wait    = frame_gap - elapsed
        if wait > 0:
            time.sleep(wait)

    cap.release()
    frame_ph.empty()
    stats_ph.empty()
    st.info("⏹️ Webcam session ended.")