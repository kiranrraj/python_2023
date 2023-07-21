import pickle, os
import cv2 as cv
import mediapipe as mp
import numpy as np


ROOT_DIR = r'D:\github\python_2023\project_experiments'
DATA_FILE = os.path.join(ROOT_DIR, "model.p")

model_dict = pickle.load(open(DATA_FILE, 'rb'))
model = model_dict['model']

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C'}
while True:

    data_aux = []
    x_arr = []
    y_arr = []

    isTrue, frame = cap.read()
    
    if isTrue:
        height, width, _ = frame.shape
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_arr.append(x)
                    y_arr.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_arr))
                    data_aux.append(y - min(y_arr))

            x1 = int(min(x_arr) * width) - 10
            y1 = int(min(y_arr) * height) - 10

            x2 = int(max(x_arr) * width) - 10
            y2 = int(max(y_arr) * height) - 10

            prediction = model.predict([np.asarray(data_aux)])
            print(int(prediction[0]))
            predicted_character = labels_dict[int(prediction[0])]

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv.putText(frame, predicted_character, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv.LINE_AA)

        cv.imshow('frame', frame)
        key = cv.waitKey(20)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()