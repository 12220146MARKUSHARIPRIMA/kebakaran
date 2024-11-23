import streamlit as st
import torch
from ultralytics import YOLO
import pygame
import tempfile
import os
from PIL import Image
import io

# Inisialisasi pygame untuk suara alarm
pygame.mixer.init()

# Fungsi untuk memutar alarm
def play_alarm():
    try:
        pygame.mixer.music.load("alrm.mp3")  # File alarm harus ada di direktori yang sama
        pygame.mixer.music.play(-1)  # -1 agar suara alarm diputar secara berulang
    except pygame.error as e:
        st.error(f"Error memutar suara alarm: {e}")

# Fungsi untuk menghentikan alarm
def stop_alarm():
    pygame.mixer.music.stop()

# Streamlit UI
st.title("Real-Time Object Detection")

# Load model YOLOv8
model_path = "best.pt"  # Pastikan file model ada di direktori yang benar
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} tidak ditemukan!")
else:
    model = YOLO(model_path)

# Tombol untuk memulai dan menghentikan deteksi
run_detection = st.button("Start Detection")
stop_detection = st.button("Stop Detection")

# Variabel untuk melacak status alarm
alarm_playing = False

# Upload gambar untuk deteksi
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)  # Membaca gambar yang diupload oleh pengguna
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Konversi gambar ke format numpy array untuk deteksi
    img_array = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    
    # Deteksi objek menggunakan YOLO
    results = model(img_array)  
    annotated_frame = results[0].plot()  # Gambar hasil deteksi
    detected = results[0].boxes

    # Jika ada objek terdeteksi dan alarm belum diputar, mainkan alarm
    if detected is not None and len(detected) > 0:
        if not alarm_playing:  # Mainkan alarm hanya jika belum diputar
            play_alarm()
            alarm_playing = True
    else:
        if alarm_playing:  # Hentikan alarm jika objek tidak terdeteksi
            stop_alarm()
            alarm_playing = False

    # Tampilkan hasil deteksi
    st.image(annotated_frame, caption="Detected Image", use_column_width=True)

    # Simpan frame ke file sementara jika diperlukan
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        annotated_frame.save(temp_file, format="JPEG")
        st.video(temp_file.name)  # Tampilkan hasil deteksi sebagai video

if stop_detection:
    st.info("Deteksi dihentikan.")
    stop_alarm()
