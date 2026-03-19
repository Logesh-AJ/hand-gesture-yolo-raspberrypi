import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize lists to store data
up_data, down_data, right_data, left_data, open_data, close_data = [], [], [], [], [], []

def save_landmarks(landmarks, label):
    # Save x, y coordinates of hand landmarks
    data = [landmark.x for landmark in landmarks.landmark] + [landmark.y for landmark in landmarks.landmark]
    if label == 'up':
        up_data.append(data)
    elif label == 'down':
        down_data.append(data)
    elif label == 'right':
        right_data.append(data)
    elif label == 'left':
        left_data.append(data)
    elif label == 'open':
        open_data.append(data)
    elif label == 'close':
        close_data.append(data)

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

    # Display the frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Use keys to save landmarks with corresponding gesture labels
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u') and result.multi_hand_landmarks:
        print("Up recorded!")
        save_landmarks(result.multi_hand_landmarks[0], 'up')
    elif key == ord('d') and result.multi_hand_landmarks:
        print("Down recorded!")
        save_landmarks(result.multi_hand_landmarks[0], 'down')
    elif key == ord('r') and result.multi_hand_landmarks:
        print("Right recorded!")
        save_landmarks(result.multi_hand_landmarks[0], 'right')
    elif key == ord('l') and result.multi_hand_landmarks:
        print("Left recorded!")
        save_landmarks(result.multi_hand_landmarks[0], 'left')
    elif key == ord('o') and result.multi_hand_landmarks:
        print("Open recorded!")
        save_landmarks(result.multi_hand_landmarks[0], 'open')
    elif key == ord('c') and result.multi_hand_landmarks:
        print("Close recorded!")
        save_landmarks(result.multi_hand_landmarks[0], 'close')
    elif key == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Save data to CSV files
np.savetxt('up_data.csv', np.array(up_data), delimiter=',')
np.savetxt('down_data.csv', np.array(down_data), delimiter=',')
np.savetxt('right_data.csv', np.array(right_data), delimiter=',')
np.savetxt('left_data.csv', np.array(left_data), delimiter=',')
np.savetxt('open_data.csv', np.array(open_data), delimiter=',')
np.savetxt('close_data.csv', np.array(close_data), delimiter=',')
