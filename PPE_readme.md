Here’s a clean and professional **README.md** for your Streamlit PPE Surveillance Platform project:  

***

# Workplace PPE Surveillance Platform  

This project is a **Streamlit-based AI-powered PPE (Personal Protective Equipment) Surveillance Platform** that leverages **YOLOv8 object detection** and **Hugging Face LLMs** to detect safety compliance in workplaces from images, videos, or webcam streams. The system automatically:  

- Detects workers' PPE (helmets, vests, gloves, etc.) using YOLO.  
- Flags safety violations (e.g., missing hard hat, missing vest).  
- Summarizes violation logs using Hugging Face text models.  
- Provides safety advice for work scenarios.  
- Sends **email alerts** with detected violations, summaries, and recommendations.  

***

## Features  
- **Custom YOLOv8 model** for detecting PPE and violations.  
- **Streamlit UI** with modern dashboard styling and background image support.  
- Supports **image upload**, **video processing**, and **real-time webcam detection**.  
- **Email alerts** are automatically sent when violations occur.  
- Integrated **Hugging Face API** for text summarization and safety advice.  
- Modular code with reusable functions for detection, logging, summarization, and alerting.  

***

## Project Structure  

```
├── weights/
│   └── best.pt              # Trained YOLO model weights
├── Images/
│   └── const_bg1.jpg         # Background image for UI
├── app.py                    # Main Streamlit application (your given code)
├── .env                      # Environment variables (email + API keys)
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

***

## Installation  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/ppe-surveillance-platform.git
   cd ppe-surveillance-platform
   ```

2. **Create & Activate Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

***

## Environment Variables  

Create a `.env` file in the project root with the following:  

```
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password      # Use App Password from Google
HF_API_TOKEN=your-huggingface-token
```

- **EMAIL_USER / EMAIL_PASS**: Used to send Gmail alerts (requires enabling App Passwords for Gmail).  
- **HF_API_TOKEN**: Hugging Face API access token (get it from https://huggingface.co/settings/tokens).  

***

## Usage  

Run the app with:  

```bash
streamlit run app.py
```

### Input Modes:  
- **Upload Image** → Detect PPE in images.  
- **Upload Video** → Detect PPE frame-by-frame in video (supports .mp4, .avi, .mov, .mkv).  
- **Webcam** → Real-time PPE surveillance with your webcam.  

### Extra Features:  
- **Summarize Detections** → Uses Hugging Face LLM to generate a short safety summary.  
- **Ask for Safety PPE Advice** → Enter a work scenario to get recommended PPE list.  
- **Email Alerts** → Sent automatically when violations are detected.  

***

## Requirements  

Add this to `requirements.txt`:  

```
streamlit
opencv-python
numpy
Pillow
ultralytics
python-dotenv
requests
```

***

## Future Enhancements  
- Add **tracking IDs** for individuals across frames.  
- Store violation logs in a **database** instead of memory.  
- Multi-user login & company-specific email alerting.  
- Support for multilingual safety advice summaries.  
- Deployment on **Streamlit Cloud / Docker / AWS / Azure**.  

***
