# Object-Detection-and-Tracking-using-Yolo-v-8-and-Deep-sort-Algorithm
Project is a Streamlit web app designed for real-time object detection and tracking. It uses the YOLOv8 model and Deep SORT algorithm to detect objects in images, videos, live camera feeds, and streamed video from a phone. The app processes these sources and displays the detected objects directly within the interface.
# Steps to set up the Project
Here's a step-by-step guide on how to build and use your Streamlit web app for real-time object detection and tracking:

### Step 1: Set Up the Environment
1. Install Python:Ensure you have Python installed on your machine.
2. Create a Virtual Environment:
   - Open your terminal or command prompt.
   - Navigate to your project directory.
   - Create a virtual environment using: `python -m venv venv`
   - Activate the environment:
     - On Windows: `venv\Scripts\activate`
     - On macOS/Linux: `source venv/bin/activate`
3. Install Required Libraries:
   - Install the necessary Python packages by running:
     ```bash
     pip install streamlit opencv-python-headless ultralytics deep_sort_realtime numpy pandas pillow
     ```

###Step 2: Build the Streamlit Web App
1. Set Up Your Project Structure:
   - Create directories for images, weights, and any other assets you need (e.g., `images/`, `weights/`).
   - Place the YOLOv8 model weights (`yolov8n.pt`) in the `weights/` directory.
   - Place any default images you want to use in the `images/` directory.

2. Write the Streamlit App Code:
   - Create a new Python file, e.g., `app.py`, in your project directory.
   - Use the code you developed to load the YOLOv8 model, initialize the Deep SORT tracker, and set up the Streamlit interface.
   - Make sure to include options to select input sources (image, video, camera, phone stream) and adjust the model confidence level.

### Step 3: Run the Streamlit App
1. Start the App:
   - In your terminal, navigate to your project directory and run:
     ```bash
     streamlit run app.py
     ```
   - This will launch the Streamlit web app in your default web browser.

### Step 4: Use the Web App
1. Select the Source:
   - Choose from the available input sources: Image, Video, Camera, or Phone Stream.
   
2. Upload or Stream Content:
   - For Image: Upload an image file (JPEG, PNG).
   - For Video: Upload a video file (MP4, AVI, etc.).
   - For Camera: Click the "Use Webcam" button to start the camera feed.
   - For Phone Stream: Enter the stream URL from your phone’s streaming app.

3. Set Confidence Level:
   - Adjust the confidence slider to set the threshold for object detection.

4. View Results:
   - The app will display the processed images or video frames with detected objects highlighted.
   - The object counts and tracking data will be displayed in the app interface.

### Step 5: Analyze Results
1. Review Detected Objects:
   - The app displays a table or list showing the tracked objects and additional data.

2. Monitor Real-Time Feeds:
   - If using the camera or phone stream, you can monitor and analyze object movement in real-time.

/////////////////////////////////////////////////////////////////////////////////////////////////////
