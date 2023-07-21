import os, pickle
import mediapipe as mp
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

ROOT_DIR = r'D:\github\python_2023\project_experiments'
DATA_DIR = os.path.join(ROOT_DIR, "Training_data")

data = []
labels = []
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

for folder in os.listdir(DATA_DIR):
    for img_file in os.listdir(os.path.join(DATA_DIR, str(folder))):
        data_aux = []
        x_arr = []
        y_arr = []
        img = cv.imread(os.path.join(DATA_DIR, folder, img_file))
        img_rbg = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        results = hands.process(img_rbg)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(img_rbg, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style(),
            #     )
                for i in range(len(hand_landmarks.landmark)):
                    x1 = hand_landmarks.landmark[i].x
                    y1 = hand_landmarks.landmark[i].y
                    x_arr.append(x1)
                    y_arr.append(y1)

                for j in range(len(hand_landmarks.landmark)):
                    x2 = hand_landmarks.landmark[j].x
                    y2 = hand_landmarks.landmark[j].y
                    data_aux.append(x2 - min(x_arr))
                    data_aux.append(y2 - min(y_arr))

        data.append(data_aux)
        labels.append(folder)


location = os.path.join(ROOT_DIR, 'data.pickle')
print(location)
f = open(location, 'wb')
pickle.dump({'data':data, 'labels':labels}, f)
f.close()
