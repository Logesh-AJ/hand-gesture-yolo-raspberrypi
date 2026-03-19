import cv2
import mediapipe as mp
import numpy as np
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)
cap = cv2.VideoCapture(0)
up = 0

down = 0
right = 0
left = 0
open_wiper = 0
close_wiper = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # extract x and y coordinates
            data = [landmark.x for landmark in hand_landmarks.landmark] + [landmark.y for landmark in hand_landmarks.landmark]
            data = np.array(data).reshape(1, -1)

            # Predict the gesture
            prediction = model.predict(data)

            if prediction == 1:
                up = 1
                down = 0
                right = 0
                left = 0
                open_wiper = 0
                close_wiper = 0
                cv2.putText(frame, "drone up", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif prediction == 0:
                up = 0
                down = 1
                right = 0
                left = 0
                open_wiper = 0
                close_wiper = 0
                cv2.putText(frame, "drone Down", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            elif prediction == 2:
                up = 0
                down = 0
                right = 1
                left = 0
                open_wiper = 0
                close_wiper = 0
                cv2.putText(frame, "drone Right", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif prediction == 3:
                up = 0
                down = 0
                right = 0
                left = 1
                open_wiper = 0
                close_wiper = 0
                cv2.putText(frame, "drone Left", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            elif prediction == 4:
                up = 0
                down = 0
                right = 0
                left = 0
                open_wiper = 1
                close_wiper = 0
                cv2.putText(frame, "drone forward", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif prediction == 5:
                up = 0
                down = 0
                right = 0
                left = 0
                open_wiper = 0
                close_wiper = 1
                cv2.putText(frame, "drone backward  ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if up == 1:
        print("drone up")
    elif down == 1:
        print("drone down")
    elif right == 1:
        print("move right")
    elif left == 1:
        print("move left")
    elif open_wiper == 1:
        print("drone forward ")
    elif close_wiper == 1:
        print("drone backward")

    cv2.imshow('Real-time Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()