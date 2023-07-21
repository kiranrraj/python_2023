# Imports 
import os, time, logging, sys
import cv2 as cv
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Constants 
ROOT_PATH = r"D:\github\python_2023\opencv_pyqt_integration" 
DATA_PATH = os.path.join(ROOT_PATH,'image') 
LOG_DIR = os.path.join(ROOT_PATH, 'log')
DETECTION_LIST = np.array(['hello', 'thanks', 'iloveyou'])
VIDEO_COUNT = 30
FRAME_COUNT = 30
START = 30

# Function to create folder
def createFolder(uri):
    if os.path.isdir(uri):
        print("Directory Found: {}".format(uri))
    else:
        try:
            os.mkdir(uri)
            print("Folder Created: {}".format(uri))
        except:
            print("Error occured when trying to create folder.")

##############  Logging Setting  ###################### 
logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("{}\\asl_detection.log".format(LOG_DIR))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

for dect_Item in DETECTION_LIST: 
    for video in range(VIDEO_COUNT):
        try: 
            dataFolder = os.path.join(DATA_PATH, dect_Item, str(video))
            os.makedirs(dataFolder)
            rootLogger.info("Data Folder Created: {}".format(dataFolder))
        except:
            rootLogger.exception("Exception occured while creating folder: {}".format(dataFolder))

def mpProcessFrame(frame, model):
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
    frameRGB.flags.writeable = False               
    processedFrame = model.process(frameRGB)            
    frameRGB.flags.writeable = True             
    frameBGR = cv.cvtColor(frameRGB, cv.COLOR_RGB2BGR) 
    return frameBGR, processedFrame


def drawLandmarks(frame, processedFrame):
    mpDrawing.draw_landmarks(frame, processedFrame.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS) 
    mpDrawing.draw_landmarks(frame, processedFrame.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
    mpDrawing.draw_landmarks(frame, processedFrame.face_landmarks, mpHolistic.FACEMESH_TESSELATION) 
    mpDrawing.draw_landmarks(frame, processedFrame.pose_landmarks, mpHolistic.POSE_CONNECTIONS)

def customStyleLandmarks(frame, processedFrame):
    mpDrawing.draw_landmarks(frame, 
                             processedFrame.left_hand_landmarks, 
                             mpHolistic.HAND_CONNECTIONS, 
                             mpDrawing.DrawingSpec(
                                    color=(121,22,76), 
                                    thickness=2, 
                                    circle_radius=4), 
                             mpDrawing.DrawingSpec(
                                    color=(121,44,250),
                                    thickness=2,
                                    circle_radius=2)
                             )  
    mpDrawing.draw_landmarks(frame, 
                             processedFrame.right_hand_landmarks, 
                             mpHolistic.HAND_CONNECTIONS, 
                             mpDrawing.DrawingSpec(
                                    color=(245,117,66), 
                                    thickness=2, 
                                    circle_radius=4), 
                             mpDrawing.DrawingSpec(
                                    color=(245,66,230), 
                                    thickness=2, 
                                    circle_radius=2)
                             ) 
    mpDrawing.draw_landmarks(frame, processedFrame.face_landmarks, mpHolistic.FACEMESH_TESSELATION, 
                             mpDrawing.DrawingSpec(
                                    color=(80,110,10), 
                                    thickness=1, 
                                    circle_radius=1), 
                             mpDrawing.DrawingSpec(
                                    color=(80,256,121), 
                                    thickness=1, 
                                    circle_radius=1)
                             ) 
    mpDrawing.draw_landmarks(frame, 
                             processedFrame.pose_landmarks, 
                             mpHolistic.POSE_CONNECTIONS,
                             mpDrawing.DrawingSpec(
                                    color=(80,22,10), 
                                    thickness=2, 
                                    circle_radius=4), 
                             mpDrawing.DrawingSpec(
                                    color=(80,44,121), 
                                    thickness=2, 
                                    circle_radius=2)
                             ) 


def getMPLandmarks(mpFrame, mpLandmark, points):
    mp_array = []
    landmark_type = getattr(mpFrame, mpLandmark)
    if landmark_type != None:
        if mpLandmark == "pose_landmarks":
            for landmarkPoint in landmark_type.landmark:
                landmarkPoint_params = ([landmarkPoint.x, 
                                         landmarkPoint.y, 
                                         landmarkPoint.z, 
                                         landmarkPoint.visibility])
                mp_array.append(landmarkPoint_params)
        else:
            for landmarkPoint in landmark_type.landmark:
                landmarkPoint_params = ([landmarkPoint.x, 
                                         landmarkPoint.y, 
                                         landmarkPoint.z])
                mp_array.append(landmarkPoint_params)        
    else:
        mp_array = np.zeros(int(points))
    return np.array(mp_array).flatten()

def concatLandmarks(mp_out):
    pose_coordinates = getMPLandmarks(mp_out, "pose_landmarks", 132)
    face_coordinates = getMPLandmarks(mp_out, "left_hand_landmarks", 63)
    left_hand_coordinates = getMPLandmarks(mp_out, "right_hand_landmarks", 63)
    right_hand_coordinates = getMPLandmarks(mp_out, "face_landmarks", 1404)
    return np.concatenate([
        pose_coordinates, 
        face_coordinates, 
        left_hand_coordinates, 
        right_hand_coordinates
    ])

createFolder(DATA_PATH)
createFolder(LOG_DIR)
mpHolistic = mp.solutions.holistic 
mpDrawing = mp.solutions.drawing_utils

capObj = cv.VideoCapture(0)

detectionMap = {label:num for num, label in enumerate(DETECTION_LIST)}
# while capObj.isOpened():
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for detectItem in DETECTION_LIST:
        for videoCount in range(VIDEO_COUNT):
            for frameCount in range(FRAME_COUNT):
            
                isTrue, webcamFrame = capObj.read()

                if not isTrue:
                    rootLogger.error("Could not get frame from webcam. Exiting...")
                    capObj.release()
                    sys.exit()
                    
                frame, pFrame = mpProcessFrame(webcamFrame, holistic)
                customStyleLandmarks(frame, pFrame)
                if frameCount == 0: 
                    cv.putText(frame, 
                               'STARTING COLLECTION', (120,200), 
                                cv.FONT_HERSHEY_SIMPLEX, 1, 
                                (0,255, 0), 4, cv.LINE_AA)
                    cv.putText(frame, 
                               'Press "r" when ready. Collecting frames for {} Video Number {}'.
                               format(detectItem, videoCount), (15,12), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (0, 0, 255), 1, cv.LINE_AA)
                    cv.imshow('OpenCV Feed', frame)
                    cv.waitKey(5000)
                    
                else: 
                    cv.putText(frame, 'Collecting frames for {} Video Number {}'.
                               format(detectItem, videoCount), (15,12), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 
                                cv.LINE_AA)
                    cv.imshow('OpenCV Feed', frame)

                mpDetectionPoints = concatLandmarks(pFrame)
                savePath = os.path.join(DATA_PATH, detectItem, str(videoCount), str(frameCount))
                np.save(savePath, mpDetectionPoints)

                if cv.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    capObj.release()
    cv.destroyAllWindows()

# vCountArr, labelsArr = [], []
# for detectItem in DETECTION_LIST:
#     for vCountItem in np.array(os.listdir(os.path.join(DATA_PATH, detectItem))).astype(int):
#         outputVid = []
#         for count in range(VIDEO_COUNT):
#             fileName = DATA_PATH, detectItem, str(vCountItem), "{}.npy".format(count)
#             loadFrame = np.load(os.path.join())
#             outputVid.append(loadFrame)
#         vCountArr.append(outputVid)
#         labelsArr.append(detectionMap[detectItem])

# X = np.array(vCountArr)
# y = tf.keras.utils.to_categorical(labelsArr).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# log_dir = os.path.join(ROOT_PATH,'Logs')
# tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# model = tf.keras.models.Sequential()
# lstmModel = tf.keras.layers.LSTM 
# denseModel = tf.keras.layers.Dense
# model.add(lstmModel(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# model.add(lstmModel(128, return_sequences=True, activation='relu'))
# model.add(lstmModel(64, return_sequences=False, activation='relu'))
# model.add(denseModel(64, activation='relu'))
# model.add(denseModel(32, activation='relu'))
# model.add(denseModel(DETECTION_LIST.shape[0], activation='softmax'))

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])