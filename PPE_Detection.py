import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import requests
import tempfile
import os
import base64

# ------ Encode local image to base64 string for background -------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Replace with your actual local image path
local_bg_img_path = r"C:\Object Detaction\Images\const_bg1.jpg"
bin_str = get_base64_of_bin_file(local_bg_img_path)

# ------ Inject CSS with base64 image -------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, #263859 60%, #17223b 100%);
        opacity: 0.95;
    }}
    .dashboard-title {{
        font-size: 2.8rem;
        font-weight: bold;
        letter-spacing: 2px;
        color: #17223b;
        background: rgba(255,255,255,0.79);
        border-radius: 10px;
        padding: 15px 30px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 6px 24px rgba(30,30,50,0.12);
    }}
    .stButton > button {{
        background-color: #17223b !important;
        color: #fff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(23,34,59,0.11);
    }}
    .stInput > div > input {{
        background: rgba(255,255,255,0.92) !important;
        border-radius: 8px !important;
        border: 1px solid #263859 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="dashboard-title">Workplace PPE Surveillance Platform</div>', unsafe_allow_html=True)

# -------------- CONFIG -----------------
MODEL_PATH = "weights/best.pt"

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# -------------- INIT SERVICES -------------
ppe_model = YOLO(MODEL_PATH)

# -------------- HUGGING FACE API -------------
HF_MODEL_SUMMARIZE = "google/flan-t5-small"  # Model supporting Inference API
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_SUMMARIZE}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def huggingface_inference(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        else:
            return "Failed to generate summary."
    except Exception as e:
        st.error(f"Hugging Face API error: {e}")
        return "Failed to get response from Hugging Face."


# -------------- FUNCTIONS -----------------
def send_email_alert(subject, body):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_USER
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Email sending failed: {e}")


def summarize_logs(objects_list):
    text_log = "\n".join(objects_list)
    prompt = f"Summarize this PPE detection log:\n{text_log}"
    return huggingface_inference(prompt)


def safety_prompt_response(user_prompt):
    prompt = f"List all the safety equipment and PPE that should be worn or used for this work scenario: {user_prompt}"
    return huggingface_inference(prompt)


def process_ppe_frame(frame, detected_objects, violations):
    results = ppe_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    if len(boxes) > 0:
        for bbox, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, bbox)
            cls_name = ppe_model.names[int(cls)]
            color = (0, 0, 255) if cls_name.startswith("NO-") else (0, 255, 0)
            if cls_name.startswith("NO-"):
                violations.append(cls_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detected_objects.append(cls_name)
    return frame, detected_objects, violations


def process_video_and_display(video_path, detected_objects, violations, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_placeholder = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed, detected_objects, violations = process_ppe_frame(frame_rgb.copy(), detected_objects, violations)
        frame_placeholder.image(processed, channels="BGR", caption=f"Frame {frame_count+1}")
        frame_count += 1
    cap.release()
    return detected_objects, violations, frame_count


# -------------- STREAMLIT UI --------------
option = st.selectbox("Choose Input", ["Upload Image", "Upload Video", "Webcam"])

detected_objects = []
violations = []

if option == "Upload Image":
    file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)
        frame = np.array(image)
        processed, detected_objects, violations = process_ppe_frame(frame.copy(), detected_objects, violations)
        st.image(processed, channels="BGR")
        if violations:
            summary = summarize_logs(violations)
            advice = safety_prompt_response(" and ".join(set(violations)))
            send_email_alert(
                "PPE Violation Detected",
                f"Missing PPE detected: {', '.join(set(violations))}\n\nSummary:\n{summary}\n\nAdvice:\n{advice}"
            )
            st.info("Violation email sent with summary and advice.")

elif option == "Upload Video":
    video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()
        st.info("Processing video, displaying frames below...")
        detected_objects, violations, frame_count = process_video_and_display(
            tfile.name, detected_objects, violations, max_frames=150)
        st.success(f"Processed {frame_count} frames.")
        os.unlink(tfile.name)
        if violations:
            summary = summarize_logs(violations)
            advice = safety_prompt_response(" and ".join(set(violations)))
            send_email_alert(
                "PPE Violation Detected",
                f"Missing PPE detected: {', '.join(set(violations))}\n\nSummary:\n{summary}\n\nAdvice:\n{advice}"
            )
            st.info("Violation email sent with summary and advice.")

elif option == "Webcam":
    run = st.checkbox("Run Webcam", key="webcam_run")

    frame_placeholder = st.empty()
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Camera error")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed, detected_objects, violations = process_ppe_frame(frame.copy(), detected_objects, violations)

        frame_placeholder.image(processed, channels="RGB")

        run = st.session_state.get("webcam_run", False)

    camera.release()

if detected_objects:
    if st.button("Summarize Detections with LLM"):
        summary = summarize_logs(detected_objects)
        st.write(summary)

st.header("Ask for Safety PPE Advice")
user_prompt = st.text_input("Describe the work or scenario (e.g. 'working at height on construction site'):")

if st.button("Get Safety PPE Advice"):
    if user_prompt.strip():
        advice = safety_prompt_response(user_prompt)
        st.write(advice)
    else:
        st.warning("Please enter a scenario or work description.")
