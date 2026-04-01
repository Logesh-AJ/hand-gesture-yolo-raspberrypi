# 🖐️ Hand Gesture Controlled System using YOLO (Raspberry Pi)

An AI-powered real-time hand gesture recognition system that uses **YOLO object detection + MediaPipe preprocessing** to control multiple system states such as engine, indicators, and lights on a Raspberry Pi using a Pi Camera.

---

## 🎥 Demo Videos

[Full Explanation ](https://youtu.be/4GUYknrspbk)


[ Demo (Without Explanation)](https://youtu.be/fc9VgeVaxB0)

---

## 🧠 Project Overview

This system captures live video using a **Pi Camera**, processes it using **MediaPipe** to isolate the hand region, and then applies a **custom-trained YOLO model** to detect gestures.

Each gesture corresponds to a control action such as:

* Engine ON/OFF
* Indicator Left / Right
* Wiper Toggle
* Low Beam / High Beam

To avoid false triggers, the system uses a **buffer-based decision logic**, ensuring gestures are stable before executing actions.

---

## 🚀 Features

* Real-time gesture detection on Raspberry Pi
* YOLO-based multi-class detection (6 gesture classes)
* MediaPipe hand region extraction
* Frame buffering for stable predictions
* Custom confidence thresholds per class
* Clean UI display for system status

---

## 📁 Project Structure

```
hand-gesture-yolo-raspberrypi/
│
├── model/                     # Trained YOLO model weights
├── model_raw_graphs&img/      # Training graphs & results
├── reports_doc/               # Project reports
├── src/                       # Main source code
│
├── demo_video.md              # Demo explanation
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🛠️ Requirements

### Hardware

* Raspberry Pi (Pi 4 / Pi 5 recommended)
* Pi Camera Module
* Monitor / SSH access

### Software

* Python 3.8+
* Raspberry Pi OS (with libcamera)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Logesh-AJ/hand-gesture-yolo-raspberrypi.git
cd hand-gesture-yolo-raspberrypi
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

⚠️ If OpenCV fails on Raspberry Pi:

```bash
pip install opencv-python-headless
```

---

### 4️⃣ Install Pi Camera Support (IMPORTANT)

```bash
sudo apt update
sudo apt install python3-picamera2
```

Also enable camera:

```bash
sudo raspi-config
```

→ Interface Options → Enable Camera

---

## ▶️ Running the Project

Navigate to source folder:

```bash
cd src
python3 main.py
```

---

## 🧠 How the System Works (Step-by-Step)

### Step 1: Video Capture

* Pi Camera captures live frames

### Step 2: Preprocessing (MediaPipe)

* Detects hand region
* Background is suppressed or blurred

### Step 3: Gesture Detection (YOLO)

* Model detects gesture class
* Bounding boxes + labels generated

### Step 4: Confidence Filtering

* Class-wise thresholds applied
* Reduces false detections

### Step 5: Buffer Logic (Stability Layer)

* Stores predictions over ~3 seconds
* Action triggered only if:

  * Same gesture appears in ≥60% frames

### Step 6: Action Execution

* System toggles corresponding state:

  * Engine
  * Indicators
  * Wiper
  * Headlights

### Step 7: Display Output

* Status window shows current system states
* Last action displayed

---

## 🎮 Gesture Mapping

| Class ID | Gesture            | Action          |
| -------- | ------------------ | --------------- |
| 0        | Fist               | Engine Toggle   |
| 1        | Two Fingers        | Indicator Right |
| 2        | Two Fingers (Left) | Indicator Left  |
| 3        | Hand Motion        | Wiper           |
| 4        | Open Palm          | Low Beam        |
| 5        | Spread Fingers     | High Beam       |

---

## 📊 Model Performance

* mAP@0.5 ≈ **0.91**
* F1 Score ≈ **0.78**
* Precision peaks near **1.0 at high confidence**

The model performs well but still has minor misclassifications under complex conditions.

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Slight delay due to buffering logic
* Limited to trained gesture classes
* Performance constrained by Raspberry Pi hardware

---

## 📈 Future Improvements

* Convert model to ONNX / TensorRT for faster inference
* Add real hardware control (GPIO integration)
* Build mobile app interface
* Expand gesture dataset
* Add voice + gesture hybrid control

---

## 🔥 Key Highlight

This project is not just gesture detection—it is a **complete real-time control system** with:

* AI model
* Preprocessing pipeline
* Stability logic
* Action execution

---

## 📌 Final Note

If you’re reviewing this project:

* Focus on the **pipeline design**, not just detection
* The strength lies in **stability + control logic**, not raw YOLO usage

---

## 👤 Author
**Logesh A J**

---
