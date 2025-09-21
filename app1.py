from pathlib import Path
import sys
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tabulate import tabulate
import base64
import pandas as pd
import plotly.express as px

# Setup paths
file_path = Path(__file__).resolve()
root_path = file_path.parent
if root_path not in sys.path:
    sys.path.append(str(root_path))
ROOT = root_path.relative_to(Path.cwd())

# Load model and tracker
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
model = YOLO(DETECTION_MODEL)
tracker = DeepSort(max_age=30)

# Streamlit page config
st.set_page_config(
    page_title="Object Detection & Tracking",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for background and styling
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
        }}
        .stButton>button {{
            background-color: #007bff;
            color: white;
            border-radius: 6px;
            height: 38px;
            width: 100%;
        }}
        table, th, td {{
            border: 1px solid white !important;
            padding: 6px;
            border-collapse: collapse;
            color: white !important;
        }}
        th {{
            background-color: rgba(0,0,0,0.7);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Background image path - update this path appropriately
add_bg_from_local("Images/object-detection-illustration.png")

# Constants
IMAGE = 'Image'
VIDEO = 'Video'
CAMERA = 'Camera'
PHONE_STREAM = 'Phone Stream'
SOURCES_LIST = [IMAGE, VIDEO, CAMERA, PHONE_STREAM]

# Analytics Data Store
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = {
        "frame_idx": [],
        "counts": []
    }

# Initialize frame counter in session state
if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0

# Function to update analytics data
def update_analytics(tracked_objects, frame_idx):
    counts_by_label = {}
    for obj in tracked_objects:
        label = obj["Label"]
        counts_by_label[label] = counts_by_label.get(label, 0) + 1
    st.session_state.analytics_data["frame_idx"].append(frame_idx)
    st.session_state.analytics_data["counts"].append(counts_by_label)

# Prepare analytics dataframe for plotting
def prepare_analytics_df():
    rows = []
    for idx, counts in zip(st.session_state.analytics_data["frame_idx"], st.session_state.analytics_data["counts"]):
        for label, count in counts.items():
            rows.append({"Frame": idx, "Label": label, "Count": count})
    df = pd.DataFrame(rows)
    return df

# Define detection and tracking function
def process_and_track(frame, model, tracker):
    results = model(frame)
    detections = results[0].boxes
    
    bboxes = []
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        confidence = detection.conf.item()
        cls_id = int(detection.cls[0])
        bboxes.append(([x1, y1, w, h], confidence, cls_id))

    tracks = tracker.update_tracks(bboxes, frame=frame)
    tracked_objects = []

    for t in tracks:
        track_id = t.track_id
        label = model.names[t.det_class]
        bbox = t.to_ltrb()
        tracked_objects.append({
            "Track ID": track_id,
            "Label": label,
            "BBox": f"{tuple(map(int, bbox))}"
        })

    annotated_frame = results[0].plot()
    return annotated_frame, tracked_objects

# Sidebar for controls and display
st.sidebar.title("Source & Controls")
source_type = st.sidebar.selectbox("Select Source", SOURCES_LIST)

# Tracker data display placeholder
tracked_object_sidebar = st.sidebar.empty()

st.title("🚀 Object Detection & Tracking Dashboard")
st.markdown("---")

# Main content container
main_container = st.container()

# Display tracked objects as styled table in sidebar
def display_tracked_table(tracked_objects):
    if tracked_objects:
        df = pd.DataFrame(tracked_objects)
        st.sidebar.markdown("### Tracked Objects")
        st.sidebar.dataframe(df, use_container_width=True)
    else:
        st.sidebar.info("No objects tracked currently.")

# Analytics plotting function with user selection
def show_analytics():
    df_analytics = prepare_analytics_df()
    if df_analytics.empty:
        st.warning("No analytics data collected yet.")
        return

    st.markdown("## Analytics")

    # Select graph type
    graph_type = st.selectbox("Select graph type", ["Bar Chart", "Line Chart", "Grouped Bar Chart"])

    if graph_type == "Bar Chart":
        # Bar chart for latest frame counts
        latest_frame = df_analytics["Frame"].max()
        fig = px.bar(df_analytics[df_analytics["Frame"] == latest_frame],
                     x="Label", y="Count",
                     title=f"Object Counts in Frame {latest_frame}",
                     color="Label")
        st.plotly_chart(fig, use_container_width=True, key=f"bar_{latest_frame}")

    elif graph_type == "Line Chart":
        # Line chart of counts over all frames
        fig = px.line(df_analytics,
                      x="Frame", y="Count", color="Label",
                      title="Object Counts Over Time")
        st.plotly_chart(fig, use_container_width=True, key="line_all")

    elif graph_type == "Grouped Bar Chart":
        # Grouped bar showing counts per label for all frames
        fig = px.bar(df_analytics,
                     x="Frame", y="Count", color="Label",
                     title="Object Counts by Frame (Grouped Bar Chart)",
                     barmode="group")
        st.plotly_chart(fig, use_container_width=True, key="grouped_bar_all")

with main_container:
    # File upload / camera source processing
    if source_type == IMAGE:
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_img, tracked_objects = process_and_track(image_array, model, tracker)

            display_tracked_table(tracked_objects)
            st.image(processed_img, caption='Processed & Tracked Image', use_column_width=True)

            # Update analytics for this single image (frame 0)
            update_analytics(tracked_objects, 0)

    elif source_type == VIDEO:
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            st.session_state.frame_counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out_frame, tracked_objects = process_and_track(frame, model, tracker)
                display_tracked_table(tracked_objects)
                stframe.image(out_frame, channels="BGR")

                update_analytics(tracked_objects, st.session_state.frame_counter)
                st.session_state.frame_counter += 1
            cap.release()

    elif source_type == CAMERA:
        def capture_camera():
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            stop_btn = st.checkbox('Stop Camera', key='stop_camera')
            st.session_state.frame_counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_btn:
                    break
                out_frame, tracked_objects = process_and_track(frame, model, tracker)
                display_tracked_table(tracked_objects)
                stframe.image(out_frame, channels="BGR")

                update_analytics(tracked_objects, st.session_state.frame_counter)
                st.session_state.frame_counter += 1
            cap.release()
        if st.button('Use Webcam'):
            capture_camera()

    elif source_type == PHONE_STREAM:
        st.markdown("### Instructions for Phone Stream")
        st.text("Use an app like IP Webcam or EpocCam, enter the stream URL below.")
        stream_url = st.text_input("Enter stream URL (e.g., http://192.168.0.100:8080/video)")
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            stframe = st.empty()
            st.session_state.frame_counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Unable to receive video from stream. Check the URL.")
                    break
                out_frame, tracked_objects = process_and_track(frame, model, tracker)
                display_tracked_table(tracked_objects)
                stframe.image(out_frame, channels="BGR")

                update_analytics(tracked_objects, st.session_state.frame_counter)
                st.session_state.frame_counter += 1
            cap.release()

# Add a button for showing analytics only on demand
if st.sidebar.button("Show Analytics"):
    show_analytics()
