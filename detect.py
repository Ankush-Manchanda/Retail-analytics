import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
from datetime import datetime
import pandas as pd
import os
import time

@st.cache_resource
def load_model():
    return YOLO("models/yolov8n.pt")

def run_detection():
    st.subheader("📸 Object Detection")

    model = load_model()
    input_type = st.radio(
        "Select Input Type",
        ["Image", "Video", "Real-Time (Webcam)"]
    )

    # ---------------- IMAGE ----------------
    if input_type == "Image":
        image_file = st.file_uploader("Upload Image", ["jpg", "jpeg", "png"])
        if image_file:
            img = Image.open(image_file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            results = model(frame)[0]
            st.image(results.plot(), channels="BGR")

    # ---------------- VIDEO ----------------
    elif input_type == "Video":
        video_file = st.file_uploader("Upload Video", ["mp4", "avi"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            log = []
            last_log = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(frame, persist=True)[0]
                annotated = results.plot()
                stframe.image(annotated, channels="BGR")

                if time.time() - last_log >= 1:
                    ids = results.boxes.id
                    count = len(set(ids.tolist())) if ids is not None else 0
                    log.append({
                        "timestamp": datetime.now(),
                        "occupancy": count
                    })
                    last_log = time.time()

            cap.release()

            df = pd.DataFrame(log)
            os.makedirs("analytics", exist_ok=True)
            df.to_csv("analytics/footfall.csv", index=False)
            st.success("📊 Video analytics saved.")

    # ---------------- REAL TIME ----------------
    elif input_type == "Real-Time (Webcam)":

        if "run" not in st.session_state:
            st.session_state.run = False
        if "cap" not in st.session_state:
            st.session_state.cap = None
        if "log" not in st.session_state:
            st.session_state.log = []

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🟢 Start Detection"):
                st.session_state.run = True
                st.session_state.cap = cv2.VideoCapture(0)
                st.session_state.log = []

        with col2:
            if st.button("🔴 Stop Detection"):
                st.session_state.run = False

        frame_box = st.empty()

        if st.session_state.run and st.session_state.cap:
            ret, frame = st.session_state.cap.read()
            if ret:
                results = model.track(frame, persist=True)[0]
                annotated = results.plot()
                frame_box.image(annotated, channels="BGR")

                ids = results.boxes.id
                count = len(set(ids.tolist())) if ids is not None else 0

                st.session_state.log.append({
                    "timestamp": datetime.now(),
                    "occupancy": count
                })

            time.sleep(0.1)
            st.rerun()

        if not st.session_state.run and st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

            if st.session_state.log:
                df = pd.DataFrame(st.session_state.log)
                os.makedirs("analytics", exist_ok=True)
                df.to_csv("analytics/footfall.csv", index=False)
                st.success("📊 Real-time analytics saved.")
                st.session_state.log = []
