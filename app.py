import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Deteksi Sampah dari Gambar",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk UI yang menarik
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #00C851 0%, #FF4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .mode-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header dengan styling
st.markdown('<p class="main-header">♻️ Deteksi Sampah Organik & Anorganik</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">YOLOv8 – Deteksi Objek dari Gambar Upload</p>', unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =============================
# SIDEBAR
# =============================
st.sidebar.markdown("---")
st.sidebar.markdown("### Mode Deteksi")
st.sidebar.info("Aplikasi ini sekarang hanya mendukung deteksi dari gambar yang diupload.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Pengaturan Deteksi")

# Penjelasan singkat
with st.sidebar.expander("Penjelasan Parameter", expanded=False):
    st.markdown("""
    **Confidence Threshold:** Tingkat kepercayaan minimum untuk deteksi.
    - Tinggi (0.7+): Hanya deteksi yang sangat yakin, kurang false positive
    - Rendah (0.1-0.3): Menangkap lebih banyak objek, bisa ada false positive
    
    **Maks. Luas Bounding Box:** Batas maksimum ukuran kotak deteksi (%).
    - Tinggi (70%+): Bisa deteksi objek besar (manusia, kendaraan)
    - Rendah (10-30%): Hanya objek kecil-menengah, fokus sampah
    """)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5, 0.05,
    help="Tingkat kepercayaan minimum untuk deteksi objek"
)

max_area_percent = st.sidebar.slider(
    "Maks. Luas Bounding Box (%)",
    10, 90, 40, 5,
    help="Batas maksimum ukuran kotak deteksi"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Informasi")
st.sidebar.info(
    "Upload Gambar: Deteksi objek dari file gambar yang diupload.\n\n"
    "Tips: Pastikan gambar jelas dan objek sampah terlihat dengan baik untuk hasil deteksi yang optimal."
)

# =============================
# HELPER FUNCTION: PROCESS DETECTION
# =============================
def process_detection(image, conf_threshold, max_area_percent):
    """Fungsi helper untuk memproses deteksi pada gambar"""
    # Convert PIL to numpy array jika perlu
    if isinstance(image, Image.Image):
        img = np.array(image)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    h, w, _ = img.shape
    max_allowed_area = (max_area_percent / 100) * (w * h)

    # Prediction
    results = model.predict(
        img,
        conf=conf_threshold,
        verbose=False
    )

    detected_objects = 0

    # Filter & Draw BBox
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        area = (x2 - x1) * (y2 - y1)
        if area > max_allowed_area:
            continue

        label = model.names[cls_id]
        text = f"{label} {conf:.2f}"

        # Warna per kelas
        if label.lower() == "organik":
            color = (0, 255, 0)  # Hijau
        else:
            color = (0, 0, 255)  # Merah

        # Hitung thickness dan font scale berdasarkan ukuran gambar
        # Font scale diperbesar lagi untuk label yang sangat jelas
        if w <= 640:
            bbox_thickness = 3
            font_scale = 1.8
            text_thickness = 3
        elif w <= 1280:
            bbox_thickness = 3
            font_scale = 2.2
            text_thickness = 3
        else:
            bbox_thickness = 4
            font_scale = 2.5
            text_thickness = 4

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, bbox_thickness)
        
        # Hitung posisi text yang lebih baik dengan background untuk readability
        # Posisi text di atas bounding box dengan padding yang disesuaikan untuk font besar
        text_offset = int(font_scale * 25)
        text_y = max(text_offset, y1 - 10) if y1 > text_offset else y1 + text_offset
        
        # Hitung ukuran text untuk background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        
        # Draw background untuk text agar lebih mudah dibaca (padding lebih besar untuk font besar)
        padding = max(5, int(font_scale * 3))
        cv2.rectangle(
            img,
            (x1, text_y - text_height - padding),
            (x1 + text_width + padding, text_y + baseline + padding),
            color,
            -1  # Filled rectangle
        )
        
        # Tentukan warna text berdasarkan label untuk kontras yang lebih baik
        # Organik (hijau) -> hitam, Anorganik (merah) -> putih
        if label.lower() == "organik":
            text_color = (0, 0, 0)  # Hitam untuk kontras dengan hijau
        else:
            text_color = (255, 255, 255)  # Putih untuk kontras dengan merah
        
        # Draw text dengan warna yang disesuaikan
        cv2.putText(
            img,
            text,
            (x1 + padding // 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness
        )

        detected_objects += 1

    return img, detected_objects


# =============================
# MAIN CONTENT AREA (Upload Gambar Saja)
# =============================
st.markdown("### Mode: Upload Gambar")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Upload File")
    uploaded_file = st.file_uploader(
        "Pilih gambar untuk dianalisis",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload gambar sampah yang ingin dideteksi",
        label_visibility="visible"
    )

with col2:
    st.markdown("#### Petunjuk")
    st.info(
        "Format yang didukung: JPG, JPEG, PNG, BMP\n\n"
        "Tips: Pastikan gambar jelas dan objek sampah terlihat dengan baik untuk hasil deteksi yang optimal."
    )

if uploaded_file:
    # Tampilkan gambar input
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Gambar Input")
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

    with col2:
        st.markdown("#### Hasil Deteksi")

        # Process detection
        with st.spinner("Memproses deteksi..."):
            img_result, detected_objects = process_detection(
                image, conf_threshold, max_area_percent
            )

        # Convert BGR to RGB for display
        img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        st.image(
            img_result_rgb,
            caption=f"Hasil Deteksi ({detected_objects} objek terdeteksi)",
            use_container_width=True
        )

        # Status message
        if detected_objects > 0:
            st.success(f"{detected_objects} objek berhasil terdeteksi")
        else:
            st.warning("Tidak ada objek yang terdeteksi. Coba ubah pengaturan confidence threshold.")

        # Download button
        st.download_button(
            label="Download Hasil Deteksi",
            data=cv2.imencode('.jpg', img_result)[1].tobytes(),
            file_name="hasil_deteksi.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
else:
    st.info("Silakan upload gambar untuk memulai deteksi.")

