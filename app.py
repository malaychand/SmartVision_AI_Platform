import streamlit as st
import os

# ✅ MUST be the very first Streamlit call
st.set_page_config(
    page_title="SmartVision",
    layout="wide",
    page_icon="👁️"
)

# ─────────────────────────────────────────────────────────────
# HERO WITH IMAGE
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0f172a, #111827);
        border-radius: 18px;
        padding: 30px 30px 40px 30px;
        margin-bottom: 28px;
        text-align: center;
    ">
    """,
    unsafe_allow_html=True,
)

# ✅ FIX 1: Use forward slash (works on Linux/Mac/Windows)
image_path = "models/smartvision.png"

# ✅ FIX 2: Check if file exists before loading to give a clear error
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning(f"⚠️ Image not found at `{image_path}`. Please ensure the file is committed to your repository.")

st.markdown(
    """
    <div style="margin-top:20px;">
        <p style="font-size:16px; color:#9ca3af; margin:0;">
            Unified Image Classification & Object Detection Platform
        </p>
        <p style="font-size:13px; color:#6b7280; margin-top:10px;">
            25-Class COCO · YOLOv8 · VGG16 · ResNet50 · MobileNetV2 · EfficientNetB0
        </p>
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("📌 Use the **sidebar** to navigate between pages.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TOP STATS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Platform Overview")

c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    ("🎯 Classes",     "25",        "#2ecc71"),
    ("🧠 CNN Models",  "4",         "#3498db"),
    ("🔍 Detection",   "YOLOv8s",   "#f39c12"),
    ("📦 Dataset",     "COCO",      "#9b59b6"),
    ("☁️ Deployment",  "HF Spaces", "#1abc9c"),
]
for col, (label, val, color) in zip([c1, c2, c3, c4, c5], stats):
    with col:
        st.markdown(
            f"""<div style="
                border:2px solid {color}55; border-radius:12px;
                padding:16px 10px; text-align:center; background:{color}0d;
            ">
                <p style="font-size:11px;color:#888;margin:0;font-weight:600;
                   letter-spacing:1px;text-transform:uppercase;">{label}</p>
                <p style="font-size:20px;font-weight:800;color:{color};margin:6px 0 0;">{val}</p>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — KEY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🚀 Key Features")

feat_col1, feat_col2 = st.columns(2)

features_left = [
    ("🖼️", "Single-object image classification",       "Upload any image — all 4 CNNs classify it simultaneously with Top-5 predictions."),
    ("📦", "Multi-object detection with bounding boxes","YOLOv8 detects and localises multiple objects in a single frame."),
    ("🎨", "Color-coded per-class visualisation",       "Each of 25 classes has a unique bounding box color for instant recognition."),
]
features_right = [
    ("📊", "Interactive performance dashboard",         "Live charts, confusion matrices, training curves, and per-class mAP breakdown."),
    ("📷", "Live webcam detection",                     "Real-time YOLOv8 inference on webcam feed with FPS overlay (local only)."),
]

for col, feats in zip([feat_col1, feat_col2], [features_left, features_right]):
    with col:
        for icon, title, desc in feats:
            st.markdown(
                f"""<div style="
                    display:flex; gap:12px; padding:12px 14px;
                    border:1.5px solid #dee2e6; border-radius:10px;
                    margin-bottom:10px; background:#fafbfc;
                ">
                    <span style="font-size:22px;flex-shrink:0;">{icon}</span>
                    <div>
                        <p style="font-size:13px;font-weight:700;margin:0 0 2px;">{title}</p>
                        <p style="font-size:12px;color:#666;margin:0;">{desc}</p>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

st.divider()

# =====================================================
# QUICK NAVIGATION BUTTONS
# =====================================================
st.header("🚀 Explore Features")

col1, col2 = st.columns(2)

with col1:
    if st.button("🖼️ Image Classification", use_container_width=True):
        st.switch_page("pages/image_classification.py")

    if st.button("📈 CNN Performance", use_container_width=True):
        st.switch_page("pages/model_performance.py")

with col2:
    if st.button("🔍 Object Detection", use_container_width=True):
        st.switch_page("pages/yolo_object_detection.py")

    if st.button("📊 YOLO Performance", use_container_width=True):
        st.switch_page("pages/yolo_performance.py")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODELS USED
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🤖 Models Used")

# ── YOLOv8 ───────────────────────────────────────────────────────────────────
st.markdown("#### 🔍 Object Detection")

yolo_col, yolo_info = st.columns([1, 3])
with yolo_col:
    st.markdown(
        """<div style="
            background:linear-gradient(135deg,#f39c1222,#f39c1211);
            border:2px solid #f39c1266; border-radius:14px;
            padding:20px; text-align:center;
        ">
            <p style="font-size:36px;margin:0;">⚡</p>
            <p style="font-size:16px;font-weight:800;color:#f39c12;margin:8px 0 2px;">YOLOv8s</p>
            <p style="font-size:11px;color:#888;margin:0;">Ultralytics</p>
        </div>""",
        unsafe_allow_html=True,
    )
with yolo_info:
    chips = [
        ("ARCHITECTURE",  "YOLOv8 Small (YOLOv8s)"),
        ("PRETRAINED ON", "COCO (80 classes)"),
        ("FINE-TUNED ON", "25-Class COCO Subset"),
        ("EPOCHS",        "50"),
        ("IMAGE SIZE",    "640 × 640"),
        ("mAP@0.5",       "0.893"),
        ("PRECISION",     "95.1%"),
        ("RECALL",        "87.7%"),
        ("INFERENCE",     "~6.7 ms · ~120 FPS"),
    ]
    chips_html = "".join(
        f'<span style="display:inline-block;background:#f8f9fa;border:1px solid #dee2e6;'
        f'border-radius:8px;padding:5px 10px;font-size:12px;margin:3px 4px 3px 0;">'
        f'<span style="color:#888;font-size:10px;display:block;font-weight:600;'
        f'letter-spacing:.5px;text-transform:uppercase;">{k}</span>'
        f'<span style="font-weight:700;color:#f39c12;">{v}</span></span>'
        for k, v in chips
    )
    st.markdown(
        f"""<div style="border:1.5px solid #f39c1233;border-radius:12px;
            padding:14px 16px;background:#f39c1208;">
            <div style="margin-bottom:10px;">{chips_html}</div>
            <p style="font-size:12px;color:#666;margin:0;line-height:1.6;">
                YOLOv8s is Ultralytics' small variant — optimised for the best balance between
                accuracy and inference speed. Fine-tuned with mosaic augmentation,
                auto-optimizer (AdamW), and AMP mixed precision.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── CNN Models ─────────────────────────────────────────────
st.markdown("#### 🧠 Image Classification — CNN Models")
CNN_MODELS = [
    {"name": "EfficientNetB0", "icon": "🟨", "color": "#f39c12", "highlight": "Best Accuracy"},
    {"name": "ResNet50",       "icon": "🟥", "color": "#e74c3c", "highlight": "Best All-Rounder"},
    {"name": "MobileNetV2",    "icon": "🟩", "color": "#2ecc71", "highlight": "Fastest Inference"},
    {"name": "VGG16",          "icon": "🟦", "color": "#3498db", "highlight": "Reliable Baseline"},
]

for model in CNN_MODELS:
    st.markdown(
        f"""
        <div style="
            padding:10px 14px;
            margin-bottom:8px;
            border-radius:10px;
            background:#f8f9fa;
            border-left:5px solid {model['color']};
        ">
            <span style="font-size:18px;">{model['icon']}</span>
            <span style="font-weight:700;font-size:15px;color:#000000;">
                {model['name']}
            </span>
            <span style="font-size:11px;margin-left:8px;color:{model['color']};font-weight:600;">
                {model['highlight']}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATASET + CLASSES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📦 Dataset")

d1, d2 = st.columns([1, 2])

with d1:
    dataset_rows = [
        ("Total Images",   "2,500"),
        ("Images / Class", "100"),
        ("Train Split",    "70%  ·  1,750 images"),
        ("Val Split",      "15%  ·  375 images"),
        ("Test Split",     "15%  ·  375 images"),
        ("Image Size",     "224 × 224 px"),
        ("Annotation",     "COCO Bounding Boxes"),
        ("Source",         "Hugging Face (streaming)"),
    ]
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 14px;border-bottom:1px solid #f0f0f0;font-size:12px;">'
        f'<span style="color:#666;font-weight:500;">{k}</span>'
        f'<span style="font-weight:700;color:#2c3e50;">{v}</span></div>'
        for k, v in dataset_rows
    )
    st.markdown(
        f'<div style="border:1.5px solid #dee2e6;border-radius:12px;overflow:hidden;">'
        f'<div style="background:#2c3e50;padding:10px 14px;">'
        f'<p style="color:#fff;font-weight:700;font-size:14px;margin:0;">COCO-25 Subset</p></div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )

with d2:
    st.markdown("**25 Supported Classes:**")

    CLASS_NAMES = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck',
        'traffic light','stop sign','bench','bird','cat','dog','horse','cow',
        'elephant','bottle','cup','bowl','pizza','cake','chair','couch','potted plant',
    ]

    ICONS = {
        'person':'🧍','bicycle':'🚲','car':'🚗','motorcycle':'🏍️','airplane':'✈️',
        'bus':'🚌','train':'🚆','truck':'🚛','traffic light':'🚦','stop sign':'🛑',
        'bench':'🪑','bird':'🐦','cat':'🐱','dog':'🐶','horse':'🐴','cow':'🐮',
        'elephant':'🐘','bottle':'🍶','cup':'☕','bowl':'🥣','pizza':'🍕',
        'cake':'🎂','chair':'🪑','couch':'🛋️','potted plant':'🪴',
    }

    badges = " ".join(
        f'<span style="display:inline-block;background:#f0f4ff;border:1px solid #c8d8f8;'
        f'border-radius:20px;padding:6px 14px;font-size:16px;margin:6px 4px;'
        f'color:#000000;font-weight:600;">'
        f'{ICONS.get(c,"•")} {c}</span>'
        for c in CLASS_NAMES
    )
    st.markdown(f'<div style="line-height:2.6;">{badges}</div>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# SECTION 5 — BUSINESS APPLICATIONS
# ─────────────────────────────────────────────────────────────
st.markdown("### 🌍 Business Applications")

APPS = [
    ("🏙️", "Smart Cities & Traffic Management",
     "Automated vehicle detection · Pedestrian safety monitoring · Parking detection · Traffic violation detection"),
    ("🛒", "Retail & E-Commerce",
     "Product recognition · Scan-free checkout · Customer behaviour analytics · Visual search"),
    ("🔒", "Security & Surveillance",
     "Intrusion detection · Unattended object alerts · Perimeter monitoring · Crowd density analysis"),
    ("🦁", "Wildlife Conservation",
     "Species identification · Habitat monitoring · Poaching prevention · Population studies"),
    ("🏥", "Healthcare",
     "PPE compliance verification · Equipment tracking · Patient fall detection · Hygiene monitoring"),
    ("🏠", "Smart Home & IoT",
     "Home automation · Security alerts · Pet activity tracking · Energy usage detection"),
    ("🌾", "Agriculture",
     "Livestock monitoring · Pest detection · Equipment tracking · Harvest readiness detection"),
    ("📦", "Logistics & Warehousing",
     "Package sorting · Inventory tracking · Quality control · Loading bay monitoring"),
]

for icon, title, description in APPS:
    st.markdown(
        f"""
        <div style="border:1.5px solid #dee2e6;border-radius:12px;padding:14px 16px;
            margin-bottom:12px;background:#fafbfc;">
            <span style="font-size:20px;">{icon}</span>
            <span style="font-size:15px;font-weight:700;color:#000;">{title}: </span>
            <span style="font-size:13px;color:#555;">{description}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────
# SECTION 6 — TECH STACK + RESULTS
# ─────────────────────────────────────────────────────────────
st.markdown("### 🛠️ Tech Stack & Project Info")

tech_col, dev_col = st.columns([1, 1])

with tech_col:
    TECH = [
        ("🐍", "Python 3.10",           "Core programming language"),
        ("🤖", "TensorFlow / Keras",     "CNN training & inference"),
        ("⚡", "Ultralytics YOLOv8",     "Object detection & fine-tuning"),
        ("👁️", "OpenCV",                "Image processing & annotation"),
        ("🌐", "Streamlit",              "Interactive web app framework"),
        ("🤗", "Hugging Face Spaces",    "Cloud deployment platform"),
        ("📊", "Matplotlib / Seaborn",   "Visualisations"),
        ("🔢", "NumPy / Pandas",         "Data processing"),
        ("🧪", "Scikit-learn",           "Evaluation metrics"),
    ]
    rows_html = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;'
        f'border-bottom:1px solid rgba(255,255,255,0.05);">'
        f'<span style="font-size:18px;width:26px;text-align:center;">{icon}</span>'
        f'<span style="font-size:13px;font-weight:600;min-width:170px;color:#e5e7eb;">{name}</span>'
        f'<span style="font-size:12px;color:#9ca3af;">{desc}</span></div>'
        for icon, name, desc in TECH
    )
    st.markdown(
        f'''<div style="border:1px solid rgba(255,255,255,0.08);border-radius:14px;
            overflow:hidden;background:linear-gradient(145deg,#0f172a,#111827);
            box-shadow:0 8px 24px rgba(0,0,0,0.4);">
            <div style="background:linear-gradient(90deg,#2563eb,#1d4ed8);padding:12px 16px;">
                <p style="color:#fff;font-weight:700;font-size:14px;margin:0;">🛠️ Tech Stack</p>
            </div>
            {rows_html}
        </div>''',
        unsafe_allow_html=True,
    )

with dev_col:
    st.markdown("""
    <div style="border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:18px 20px;
        background:linear-gradient(145deg,#0f172a,#111827);box-shadow:0 8px 24px rgba(0,0,0,0.4);
        color:#e5e7eb;line-height:1.9;font-size:14px;">
        <p style="font-weight:700;font-size:16px;margin-bottom:12px;">📊 Model Scope & Results</p>
        Image Classification (Transfer Learning)<br>
        Object Detection (YOLOv8 Fine-tuning)<br>
        Evaluation — Accuracy · mAP@0.5 · Precision · Recall · F1<br><br>
        <span style="color:#22c55e;font-weight:600;">CNN Result — 92% (EfficientNetB0) ✅</span><br>
        <span style="color:#22c55e;font-weight:600;">YOLO Result — mAP@0.5 = 0.893 ✅</span>
    </div>
    """, unsafe_allow_html=True)