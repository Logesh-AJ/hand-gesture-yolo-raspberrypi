#toggle by detection an particular fram 
#to capture the frame press space 
#to close press q 

from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Suppress YOLO logs
sys.stdout = open(os.devnull, 'w')
model = YOLO(r'C:\Users\logesh A.J\Documents\projects\hand guester control (with neethu mam )\try_3\files\best.pt', verbose=False)
sys.stdout = sys.__stdout__

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

gesture_actions = {
    0: "Engine Toggle",
    1: "Indicator Right",
    2: "Indicator Left",
    3: "Wiper",
    4: "Headlight Low Beam",
    5: "Headlight High Beam"
}

class_conf_thresholds = {
    0: 0.5,
    1: 0.5,
    2: 0.5,
    3: 0.1,
    4: 0.8,
    5: 0.7
}

# Status variables
engine_on = False
indicator_right = False
indicator_left = False
wiper_on = False
low_beam_on = False
high_beam_on = False
recent_message = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = hands.process(rgb)
    black_bg = np.zeros_like(frame)
    mask = np.zeros_like(frame)

    if results_mp.multi_hand_landmarks:
        for hand_landmarks in results_mp.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            x_min = max(x_min - 20, 0)
            y_min = max(y_min - 20, 0)
            x_max = min(x_max + 20, w)
            y_max = min(y_max + 20, h)

            mask[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]

    processed_frame = np.where(mask != 0, mask, black_bg)

    annotated_frame = processed_frame.copy()
    cv2.putText(annotated_frame, "Press SPACE to capture gesture", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Detection', annotated_frame)

    # Create status window
    status_screen = np.ones((400, 400, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(status_screen, f"Engine: {'ON' if engine_on else 'OFF'}", (10, 50), font, 0.7, (0, 0, 0), 2)
    cv2.putText(status_screen, f"Indicator Right: {'ON' if indicator_right else 'OFF'}", (10, 90), font, 0.7, (0, 0, 0), 2)
    cv2.putText(status_screen, f"Indicator Left: {'ON' if indicator_left else 'OFF'}", (10, 130), font, 0.7, (0, 0, 0), 2)
    cv2.putText(status_screen, f"Wiper: {'ON' if wiper_on else 'OFF'}", (10, 170), font, 0.7, (0, 0, 0), 2)
    cv2.putText(status_screen, f"Low Beam: {'ON' if low_beam_on else 'OFF'}", (10, 210), font, 0.7, (0, 0, 0), 2)
    cv2.putText(status_screen, f"High Beam: {'ON' if high_beam_on else 'OFF'}", (10, 250), font, 0.7, (0, 0, 0), 2)
    cv2.putText(status_screen, "Last Msg:", (10, 310), font, 0.7, (0, 0, 0), 2)
    cv2.putText(status_screen, recent_message, (10, 350), font, 0.7, (0, 0, 255), 2)

    cv2.imshow("Car Status", status_screen)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        sys.stdout = open(os.devnull, 'w')
        results = model(processed_frame, conf=0.3)
        sys.stdout = sys.__stdout__

        boxes = results[0].boxes
        detected = False
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                required_conf = class_conf_thresholds.get(cls, 0.5)
                if conf >= required_conf:
                    action = gesture_actions.get(cls, "Unknown Gesture")
                    if cls == 0:
                        engine_on = not engine_on
                        recent_message = f"Engine {'ON' if engine_on else 'OFF'}"
                    elif cls == 1:
                        indicator_right = not indicator_right
                        recent_message = "Indicator Right Toggled"
                    elif cls == 2:
                        indicator_left = not indicator_left
                        recent_message = "Indicator Left Toggled"
                    elif cls == 3:
                        wiper_on = not wiper_on
                        recent_message = "Wiper Toggled"
                    elif cls == 4:
                        low_beam_on = not low_beam_on
                        recent_message = "Low Beam Toggled"
                    elif cls == 5:
                        high_beam_on = not high_beam_on
                        recent_message = "High Beam Toggled"
                    detected = True
                    break
        if not detected:
            recent_message = "No gesture detected"

cap.release()
cv2.destroyAllWindows()
