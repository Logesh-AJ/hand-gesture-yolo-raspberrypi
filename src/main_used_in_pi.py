from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque, Counter
from picamera2 import Picamera2
from gpiozero import LED

# ================= CONFIG =================
MODEL_PATH = "/home/proton/Downloads/files/best.pt"

class_conf_thresholds = {
    0: 0.1,
    1: 0.1,
    2: 0.1,
    3: 0.3,
    4: 0.7,
    5: 0.7
}

PROCESS_EVERY_N_FRAMES = 3
MODEL_INPUT_SIZE = 320
GESTURE_TIME = 1.5   # 🔥 Faster response

# ================= GPIO SETUP =================
ENGINE_PIN = LED(17)
RIGHT_INDICATOR_PIN = LED(27)
LEFT_INDICATOR_PIN = LED(22)
WIPER_PIN = LED(23)
LOW_BEAM_PIN = LED(24)
HIGH_BEAM_PIN = LED(25)

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= CAMERA SETUP =================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'BGR888', "size": (640, 480)}
))
picam2.start()

# ================= STATES =================
engine_on = False
indicator_right = False
indicator_left = False
wiper_on = False
low_beam_on = False
high_beam_on = False

recent_message = ""

gesture_buffer = deque(maxlen=60)
start_time = time.time()

frame_count = 0
last_boxes = None

# ================= MAIN LOOP =================
while True:
    frame = picam2.capture_array()

    # Crop square
    h, w, _ = frame.shape
    size = min(h, w)
    x = (w - size) // 2
    y = (h - size) // 2
    frame = frame[y:y+size, x:x+size]

    # Resize for speed
    frame_small = cv2.resize(frame, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

    frame_count += 1

    # Run YOLO every N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        results = model(frame_small, imgsz=MODEL_INPUT_SIZE, conf=0.3, verbose=False)
        last_boxes = results[0].boxes

    boxes = last_boxes

    # ===== CLASS FILTER =====
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            required_conf = class_conf_thresholds.get(cls, 0.5)

            if conf >= required_conf:
                gesture_buffer.append(cls)
    else:
        gesture_buffer.append(None)

    # ===== TOGGLE LOGIC (1.5 sec window) =====
    if time.time() - start_time >= GESTURE_TIME:
        counts = Counter(gesture_buffer)
        gesture_id, count = counts.most_common(1)[0] if counts else (None, 0)

        percent = (count / len(gesture_buffer)) * 100 if len(gesture_buffer) > 0 else 0

        if gesture_id is not None and percent >= 60:

            if gesture_id == 0:
                engine_on = not engine_on
                recent_message = "Engine"

            elif gesture_id == 1:
                indicator_right = not indicator_right
                recent_message = "Right Indicator"

            elif gesture_id == 2:
                indicator_left = not indicator_left
                recent_message = "Left Indicator"

            elif gesture_id == 3:
                wiper_on = not wiper_on
                recent_message = "Wiper"

            elif gesture_id == 4:
                low_beam_on = not low_beam_on
                recent_message = "Low Beam"

            elif gesture_id == 5:
                high_beam_on = not high_beam_on
                recent_message = "High Beam"

        gesture_buffer.clear()
        start_time = time.time()

    # ===== GPIO SYNC (REAL WORLD OUTPUT) =====
    ENGINE_PIN.value = engine_on
    RIGHT_INDICATOR_PIN.value = indicator_right
    LEFT_INDICATOR_PIN.value = indicator_left
    WIPER_PIN.value = wiper_on
    LOW_BEAM_PIN.value = low_beam_on
    HIGH_BEAM_PIN.value = high_beam_on

    # ===== DRAW BOUNDING BOXES =====
    display_frame = frame_small.copy()

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            color = (0, 255, 0)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame,
                        f"{cls}:{conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1)

    display_frame = cv2.resize(display_frame, (480, 480))
    cv2.imshow("Gesture Detection", display_frame)

    # ===== DASHBOARD UI =====
    status_screen = np.zeros((400, 500, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    rows = [
        ("ENGINE", engine_on),
        ("IND RIGHT", indicator_right),
        ("IND LEFT", indicator_left),
        ("WIPER", wiper_on),
        ("LOW BEAM", low_beam_on),
        ("HIGH BEAM", high_beam_on)
    ]

    y = 60
    for name, state in rows:
        color = (0, 255, 0) if state else (100, 100, 100)
        status = "ON" if state else "OFF"

        cv2.putText(status_screen, name, (30, y), font, 0.7, (200,200,200), 2)
        cv2.putText(status_screen, status, (300, y), font, 0.7, color, 2)
        y += 45

    cv2.putText(status_screen, "LAST ACTION:", (30, y+20),
                font, 0.7, (150,150,150), 2)
    cv2.putText(status_screen, recent_message, (30, y+60),
                font, 0.8, (0, 0, 255), 2)

    cv2.imshow("Car Status", status_screen)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()