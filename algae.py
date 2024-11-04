import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time

# Load the YOLOv8 model
model = YOLO(r'C:\Users\supriya janjirala\Desktop\AGRI_RASPBERRY\Algae_Detection\best.pt')

# Directory to store captured images
capture_dir = "captured_images"
os.makedirs(capture_dir, exist_ok=True)

# Function to detect algae in an image
def detect_algae_image(image):
    # Run YOLO model on the uploaded image
    results = model(image)

    # Get the bounding boxes and confidence score
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Coordinates of the bounding box
            score = box.conf.item()  # Confidence score
            class_id = box.cls.item()  # Class ID

            if score > 0.5:  # Apply confidence threshold
                # Draw the bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f'Algae: {score:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Function to capture images every 2 seconds during webcam detection
def capture_images_sequence(frame, sequence_counter):
    # Save the frame as an image
    image_path = os.path.join(capture_dir, f"algae_capture_{sequence_counter}.jpg")
    cv2.imwrite(image_path, frame)
    return image_path

# Function to detect algae from webcam with periodic capture
def detect_algae_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    stframe = st.empty()
    
    # Create the stop button outside the loop
    stop_button_pressed = st.button("Stop Camera", key="stop_camera")
    
    # Initialize variables for capturing images
    sequence_counter = 0
    captured_images = []

    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        # Run detection on the frame
        results = detect_algae_image(frame)

        # Display the frame with detection boxes
        stframe.image(results, channels="BGR", use_column_width=True)

        # Capture an image every 2 seconds
        if time.time() - last_capture_time >= 2:
            image_path = capture_images_sequence(results, sequence_counter)
            captured_images.append(image_path)
            sequence_counter += 1
            last_capture_time = time.time()

        # Check if 'Stop Camera' button is pressed
        if stop_button_pressed:
            break

    cap.release()

    return captured_images

# Streamlit App layout
st.sidebar.title("Algae Detection")
app_mode = st.sidebar.selectbox("Choose Mode", ["Predict Image", "Real-Time Detection", "View Captured Sequence"])

# Mode: Predict Image from Upload
if app_mode == "Predict Image":
    st.title("Algae Detection with YOLOv8")

    # Section to upload an image for detection
    st.header("Upload an Image for Algae Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded image to an OpenCV image
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Algae"):
            # Run algae detection on the uploaded image
            output_image = detect_algae_image(image)
            st.image(output_image, caption="Algae Detection Result", use_column_width=True)

# Mode: Real-Time Detection from Webcam
elif app_mode == "Real-Time Detection":
    st.title("Real-Time Algae Detection")

    # Section to start real-time algae detection via webcam
    st.header("Real-time Algae Detection via Camera")
    if st.button("Start Camera Detection"):
        st.write("Starting camera... press 'Stop Camera' to end.")
        captured_images = detect_algae_webcam()

        # Store captured images
        if captured_images:
            st.write(f"{len(captured_images)} images captured during detection.")

# Mode: View Captured Image Sequence
elif app_mode == "View Captured Sequence":
    st.title("View Captured Image Sequence")

    # Get list of captured images
    captured_images = [f for f in os.listdir(capture_dir) if f.endswith(".jpg")]

    if captured_images:
        # Create a dropdown to select an image from the captured sequence
        selected_image = st.selectbox("Select an image to view", captured_images)

        if selected_image:
            # Display the selected image
            image_path = os.path.join(capture_dir, selected_image)
            st.image(image_path, caption=f"Captured Image: {selected_image}", use_column_width=True)
    else:
        st.write("No images have been captured yet.")
