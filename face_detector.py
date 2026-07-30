import cv2
import streamlit as st
import os

# 1. PAGE CONFIGURATION & HEADER

st.set_page_config(page_title="Smart Face & Feature Detector", layout="wide")

st.title("📷 Real-Time Face, Eye & Smile Detection")
st.caption("Powered by OpenCV Haar Cascades and Streamlit UI Integration")


# 2. SIDEBAR CONTROLS

st.sidebar.header("⚙️ Control Settings")
detect_eyes = st.sidebar.checkbox("Detect Eyes", value=True)
detect_smiles = st.sidebar.checkbox("Detect Smiles", value=True)
detect_glasses = st.sidebar.checkbox("Detect Eyeglasses", value=True)

stop_button = st.sidebar.button("⏹️ Stop Camera", type="primary")


# 3. LOAD CASCADE CLASSIFIERS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, r"haarcascade_frontalface_alt.xml"))
eye_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR,r"haarcascade_eye.xml"))
smile_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR,r"haarcascade_smile.xml"))
eye_glasses = cv2.CascadeClassifier(os.path.join(BASE_DIR,r"haarcascade_eye_tree_eyeglasses.xml"))


# 4. WEBCAM LOOP & DETECTION LOGIC

frame_window = st.image([])
capture = cv2.VideoCapture(0)

try:
    while capture.isOpened() and not stop_button:
        ret, frame = capture.read()
        if not ret:
            st.error("Webcam video feed not accessible!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
        # Green bounding box for Face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

        # Eye Detection
            if detect_eyes:
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 30)
                if len(eyes) > 0:
                    cv2.putText(frame, "Eyes Detected", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # Smile Detection
            if detect_smiles:
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 55)
                if len(smiles) > 0:
                    cv2.putText(frame, "Smile Detected", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Eyeglasses Detection
            if detect_glasses:
                eyes_tree = eye_glasses.detectMultiScale(roi_gray, 1.5, 30)
                if len(eyes_tree) > 0:
                    cv2.putText(frame, "Eyeglasses Detected", (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)
finally:
    capture.release()
    cv2.destroyAllWindows()