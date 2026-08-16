# 🚦 Traffic Sign Recognition with YOLO

A computer vision and deep learning project for real-time **Traffic Sign and Traffic Signal Recognition** using YOLO.

🔗 **Kaggle Notebook**: [Traffic Sign Recognition by Md Mishkatul Masabi](https://www.kaggle.com/code/mdmishkatulmasabi/traffic-sign-recognition)

---

## 📌 Overview

This repository provides an end-to-end pipeline for detecting and classifying traffic signs and traffic lights in images, videos, and real-time camera streams using a custom-trained YOLO model (`best.pt`).

### 🏷️ Supported Classes (15 Classes)
- 🟢 **Green Light**
- 🔴 **Red Light**
- 🛑 **Stop**
- 🔢 **Speed Limit Signs**: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 km/h

---

## 📁 Repository Structure

```
├── best.pt              # Trained YOLO model weights
├── predict.py           # Inference script for images, videos, and live webcam
├── requirements.txt     # Python package dependencies
├── .gitignore           # Git ignore rules (excludes videos and output files)
└── README.md            # Project documentation & Kaggle notebook link
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/mmiskatul/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Create and activate virtual environment
**PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Command Prompt / Linux / macOS:**
```bash
# Windows cmd:
.venv\Scripts\activate.bat

# Linux / macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run Inference

### 1. Run on a single image
```bash
python predict.py --source "path/to/image.jpg"
```

### 2. Run on a video file
```bash
python predict.py --source "path/to/video.mp4" --project "output" --name "video_results"
```

### 3. Run on a live webcam stream
```bash
python predict.py --source 0 --show
```

### 4. Adjust confidence threshold (e.g. 0.40)
```bash
python predict.py --source "path/to/image.jpg" --conf 0.40
```

---

## 💻 Programmatic Usage in Python

```python
from ultralytics import YOLO

# Load model weights
model = YOLO("best.pt")

# Predict on image or video
results = model.predict(source="image.jpg", conf=0.25, save=True)

# Parse detections
for r in results:
    for box in r.boxes:
        class_name = model.names[int(box.cls[0].item())]
        confidence = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()
        print(f"Detected {class_name} with {confidence:.2%} confidence at {coords}")
```

---

## 🔗 Links & Attribution
- **Kaggle Notebook**: [Traffic Sign Recognition](https://www.kaggle.com/code/mdmishkatulmasabi/traffic-sign-recognition)
- **Author**: Md Mishkatul Masabi
