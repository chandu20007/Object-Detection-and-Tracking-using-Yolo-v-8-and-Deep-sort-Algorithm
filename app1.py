from pathlib import Path
import sys
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import sqlite3

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
CAMERA = 'Camera'
PHONE_STREAM = 'Phone Stream'
SOURCES_LIST = [IMAGE, VIDEO, CAMERA, PHONE_STREAM]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'

# Load model
model = YOLO(DETECTION_MODEL)

# Database file
DB_PATH = ROOT / 'object_detections.db'

# Initialize the database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop the table if it exists and recreate it without the 'shape' column
    cursor.execute('DROP TABLE IF EXISTS detections')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object TEXT NOT NULL,
            date TEXT NOT NULL,
            dimensions TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Save detected objects to the database
def save_to_db(detected_objects):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get current date and time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d %H:%M:%S")
    
    for obj in detected_objects:
        cursor.execute('''
            INSERT INTO detections (object, date, dimensions)
            VALUES (?, ?, ?)
        ''', (obj['name'], current_date, obj['dimensions']))
    
    conn.commit()
    conn.close()

# Retrieve data from the database
def get_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    query = 'SELECT * FROM detections'
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Function to process frame with YOLO model
def process_frame(frame, model, confidence):
    results = model(frame)
    detections = results[0].boxes  # Get bounding boxes
    annotated_frame = results[0].plot()  # Annotate frame with detections
    detected_objects = []
    
    for detection in detections:
        if detection.conf.item() >= confidence / 100.0:
            cls = int(detection.cls[0])
            label = model.names[cls]  # Get label of detected object
            x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])  # Get bounding box coordinates
            width = x_max - x_min
            height = y_max - y_min
            dimensions = f"{width}x{height}"
            
            detected_objects.append({
                "name": label,
                "dimensions": dimensions
            })
    
    return annotated_frame, detected_objects

# Custom CSS for Streamlit
st.markdown(
    """
    <style>
    .full-width {
        width: 100%;
        max-width: 1200px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit application
st.title("Object Detection and Tracking")
st.markdown("---")

# Model confidence slider
confidence = st.slider("Select Model Confidence", 25, 100, 85)

# Source selection dropdown
source_type = st.selectbox("Select Source", SOURCES_LIST)

# Handle different sources
if source_type == IMAGE:
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_image, detected_objects = process_frame(image, model, confidence)
        st.image(processed_image, caption='Processed Image', use_column_width=True)
        save_to_db(detected_objects)

elif source_type == VIDEO:
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        video_detected_objects = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, detected_objects = process_frame(frame, model, confidence)
            video_detected_objects.extend(detected_objects)
            stframe.image(frame, channels="BGR")
        
        cap.release()
        save_to_db(video_detected_objects)

elif source_type == CAMERA:
    # Function to capture video from webcam
    def capture_camera():
        cap = cv2.VideoCapture(0)  # 0 for default camera
        stframe = st.empty()
        camera_detected_objects = []

        # Use a Streamlit checkbox to stop the camera
        stop_button = st.checkbox('Stop Camera', key='stop_camera')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break
            frame, detected_objects = process_frame(frame, model, confidence)
            camera_detected_objects.extend(detected_objects)
            stframe.image(frame, channels="BGR")
        
        cap.release()
        save_to_db(camera_detected_objects)

    # Streamlit interface for camera capture
    if st.button('Use Webcam', key='use_webcam'):
        capture_camera()

elif source_type == PHONE_STREAM:
    st.markdown("### Instructions for Phone Stream")
    st.text("1. Use an app like IP Webcam (Android) or EpocCam (iOS) to stream video.")
    st.text("2. Connect your phone and computer to the same Wi-Fi network.")
    st.text("3. Enter the stream URL below.")

    stream_url = st.text_input("Enter the stream URL (e.g., http://192.168.0.101:8080/video)")

    if stream_url:
        cap = cv2.VideoCapture(stream_url)
        stframe = st.empty()
        phone_detected_objects = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Unable to receive video from the stream. Please check the URL and try again.")
                break
            frame, detected_objects = process_frame(frame, model, confidence)
            phone_detected_objects.extend(detected_objects)
            stframe.image(frame, channels="BGR")

        cap.release()
        save_to_db(phone_detected_objects)

# Display data from the database
st.markdown("---")
st.subheader("Detected Objects Data")
data_df = get_data_from_db()
st.dataframe(data_df)
