# 🎥 Demo Videos – Hand Gesture Controlled System (Raspberry Pi)

## 🔗 Full Explanation

https://youtu.be/4GUYknrspbk

## 🔗 Demo (Without Explanation)

https://youtu.be/fc9VgeVaxB0

---

## 🧠 Overview

This project demonstrates a **real-time hand gesture control system** running on a Raspberry Pi with a Pi Camera. It uses **YOLO-based object 
detection** combined with **MediaPipe preprocessing** to recognize gestures and control system states.

---

## ⚙️ What the Demo Shows

* Real-time hand gesture detection using Pi Camera
* YOLO model identifying gesture classes
* MediaPipe isolating hand region for better accuracy
* Stable detection using buffered decision logic
* System state changes based on gestures

---

## 🎮 Controls Demonstrated

* Engine ON/OFF
* Indicator Left / Right
* Wiper Toggle
* Low Beam / High Beam

(Each action is triggered only after consistent detection across multiple frames to avoid noise.)

---

## 🧪 Key Technical Highlights

* Custom-trained YOLO model (~6 gesture classes) 
* ~8000 image dataset with labeled gestures 
* mAP ≈ 0.91 indicating strong detection performance 
* Confidence-based filtering and class-wise thresholds
* Frame buffering (≈3 sec window) for stable execution

---

## 🎯 Key Outcome

The system successfully enables **gesture-based control of multiple functions in real time**, demonstrating a practical human-machine interaction prototype.

---

## ⚠️ Limitations

* Performance depends on lighting conditions
* Slight delay due to stability buffering
* Limited to trained gesture classes
* Not optimized for low-power edge deployment
