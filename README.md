# Object Detection and Tracking Dashboard

## Overview
This project is a real-time object detection and tracking web application built using Streamlit, YOLOv8, and DeepSort. The dashboard supports image, video, webcam, and phone stream sources for object detection and tracking. It offers a modern, user-friendly interface with a customizable background and a sidebar for controls, including source selection and tracked objects display.

An on-demand analytics feature includes interactive charts and graphs for analyzing detected object counts over time, helping visualize detections and track trends effectively.

## Features
- Real-time object detection using the YOLOv8 model.
- Object tracking with DeepSort, including unique track IDs.
- Supports multiple input sources:
  - Image upload
  - Video upload
  - Webcam live feed
  - Phone camera stream via URL
- Clean, attractive dashboard design with custom background and theming.
- Sidebar with control widgets for source selection and displaying tracked object details.
- On-demand analytics panel accessible from sidebar:
  - Bar charts of object counts per frame
  - Line charts showing count trends across frames
  - Grouped bar charts for detailed per-frame analysis
- Analytics visualize only previously detected objects and can be toggled by user.
- Responsive layout for smooth user experience.

## Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (recommended for YOLOv8 real-time performance)
- Git (for cloning repo)

### Required Python Packages
Install dependencies via pip:

```bash
pip install streamlit opencv-python ultralytics deep_sort_realtime pillow pandas plotly tabulate
```

- `streamlit`: Web app framework.
- `opencv-python`: Image and video processing.
- `ultralytics`: YOLOv8 model and inference.
- `deep_sort_realtime`: DeepSort tracking implementation.
- `pillow`: Image handling.
- `pandas`: Data manipulation for analytics.
- `plotly`: Interactive plotting.
- `tabulate`: Terminal formatted tables (optional for debugging).

## Usage

1. Clone the repository:

```bash
git clone <repo-url>
cd <repo-folder>
```

2. Place your YOLOv8 weights file (e.g., `yolov8n.pt`) inside `weights/` directory as per project structure.

3. Add a background image inside the `Images/` folder or update the path in the code to your preferred image.

4. Run the Streamlit app:

```bash
streamlit run object_detection.py
```

5. Use the sidebar to select the input source:
   - Upload images or videos for detection.
   - Use live webcam feed.
   - Connect to a phone stream by entering its URL.

6. Perform detections and see tracking results visualized with bounding boxes and labels.

7. Click the **"Show Analytics"** button in the sidebar to view interactive charts analyzing detected objects over time.

## Project Structure

```
├── Images/
│   └── object-detection-illustration.png   # Background image for the dashboard
├── weights/
│   └── yolov8n.pt                          # YOLOv8 model weights
├── object_detection.py                     # Main Streamlit app script
├── README.md                              # This file
└── requirements.txt                       # (Optional) Python package list
```

## Notes

- For best performance, run on a machine with a CUDA-enabled GPU.
- Analytics data accumulate during the session; restarting the app clears data.
- Phone Stream requires a compatible streaming app (like IP Webcam or EpocCam) running on the phone.
- Adjust the background image path in the script as needed.
- Use stable internet and camera device drivers for smoother streaming.

## Future Enhancements

- Add filters and search for tracked objects in sidebar.
- Export tracking data and analytics charts as CSV or images.
- Add alert/notification system for particular object detections.
- Performance optimization for longer video and stream sessions.
- Multi-camera support and customizable tracking parameters.

## License
This project is open-source and available under the MIT License.

***

This project showcases integrating state-of-the-art object detection and tracking algorithms into an interactive web dashboard with useful analytics, ideal for surveillance, smart monitoring, and research purposes.
