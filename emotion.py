import cv2
from deepface import DeepFace
import streamlit as st
from PIL import Image
import numpy as np

# Streamlit page configuration
st.title("Real-time Emotion Detection")
st.write("This app detects emotions in real-time from your webcam feed.")

# Sidebar controls
start_video = st.button("Start Video")
stop_video = st.button("Stop Video", key="stop_video")

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function for real-time emotion detection
def emotion_detection():
    cap = cv2.VideoCapture(0)  # Open the webcam
    stframe = st.empty()  # Placeholder for the video frame

    while start_video and not stop_video:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except:
                emotion = "Unknown"

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)  # Updated line

    cap.release()


# Run emotion detection if button is clicked
if start_video:
    emotion_detection()
