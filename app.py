import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from playsound import playsound
import threading
import tempfile

# Fungsi untuk memutar alarm
def play_alarm():
    threading.Thread(target=playsound, args=("alrm.mp3",), daemon=True).start()

# Streamlit UI
st.title("Real-Time Object Detection")

# Load model YOLOv8
model = YOLO("best.pt")  # Ganti dengan path file model Anda

# Tombol untuk memulai deteksi
run_detection = st.button("Start Detection")  
stop_detection = st.button("Stop Detection")  

# Temp file untuk video
temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')

if run_detection:
    cap = cv2.VideoCapture(0)  # Buka kamera
    if not cap.isOpened():
        st.error("Kamera tidak dapat diakses!")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_video.name, fourcc, 20.0, (640, 480))

        stframe = st.empty()  # Placeholder untuk video

        # Variabel untuk melacak status alarm
        alarm_playing = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Tidak dapat membaca frame dari kamera.")
                break

            # Deteksi objek
            results = model(frame)
            annotated_frame = results[0].plot()  # Gambar hasil deteksi
            detected = results[0].boxes

            # Jika ada objek terdeteksi dan alarm belum diputar, mainkan alarm
            if detected is not None and len(detected) > 0:
                if not alarm_playing:  # Mainkan alarm hanya jika belum diputar
                    play_alarm()
                    alarm_playing = True
            else:
                alarm_playing = False

            # Tampilkan hasil di Streamlit
            stframe.image(annotated_frame, channels="BGR")

            # Simpan frame ke video
            out.write(annotated_frame)

        cap.release()
        out.release()

if stop_detection:
    st.info("Deteksi dihentikan.")

st.video(temp_video.name)
