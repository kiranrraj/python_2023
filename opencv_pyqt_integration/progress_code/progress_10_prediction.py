import os, time, sys, logging
import cv2 as cv                    
import numpy as np                  
import mediapipe as mp
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

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

# Constants 
ROOT_PATH = r"D:\github\python_2023\opencv_pyqt_integration" 
DATA_PATH = os.path.join(ROOT_PATH,'image') 
LOG_DIR = os.path.join(ROOT_PATH, 'log')
DETECTION_LIST = np.array(['hello', 'thanks', 'iloveyou'])
VIDEO_COUNT = 30
FRAME_COUNT = 30

label_map = {label:num for num, label in enumerate(DETECTION_LIST)}

createFolder(DATA_PATH)
createFolder(LOG_DIR)

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

mpHolistic = mp.solutions.holistic 
mpDrawing = mp.solutions.drawing_utils
capObj = cv.VideoCapture(0)

# Function to create folder
for dect_Item in DETECTION_LIST: 
    for video in range(VIDEO_COUNT):
        dataFolder = os.path.join(DATA_PATH, dect_Item, str(video))
        try: 
            if os.path.isdir(dataFolder):
                print("Folder exists, {} not created".format(dataFolder))
                continue
            else:
                os.makedirs(dataFolder)
                print("Data Folder Created: {}".format(dataFolder))
        except:
            print("Exception occured while creating folder: {}".format(dataFolder))

# Mediapipe detection function
def mpProcessFrame(frame, model):
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
    frameRGB.flags.writeable = False    
    processedFrame = model.process(frameRGB)            
    frameRGB.flags.writeable = True     
    frameBGR = cv.cvtColor(frameRGB, cv.COLOR_RGB2BGR) 
    return frameBGR, processedFrame

# Custom drawing for landmarks. The function take output of the mediapipe processed image and result
def drawLandmarksCustom(frame, processedFrame):
    mpDrawing.draw_landmarks(frame, 
                             processedFrame.left_hand_landmarks, 
                             mpHolistic.HAND_CONNECTIONS, 
                             mpDrawing.DrawingSpec(         # Dot Properties
                                    color=(121,22,76), 
                                    thickness=1, 
                                    circle_radius=1), 
                             mpDrawing.DrawingSpec(         # Line Properties
                                    color=(121,44,250),
                                    thickness=2,
                                    circle_radius=1)
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
    mpDrawing.draw_landmarks(frame, processedFrame.face_landmarks, 
                             mpHolistic.FACEMESH_TESSELATION, 
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
    
# Get landmark key points and put it into a array
def getMPLandmarks(mpFrame, mpLandmark, points):
    mpKpLandmarkArr = []
    landmark_type = getattr(mpFrame, mpLandmark)
    if landmark_type != None:
        if mpLandmark == "pose_landmarks":
       #      Only pose landmark have the visibility property
            for landmarkPoint in landmark_type.landmark:
                landmarkPoint_params = ([landmarkPoint.x, 
                                         landmarkPoint.y, 
                                         landmarkPoint.z, 
                                         landmarkPoint.visibility])
                mpKpLandmarkArr.append(landmarkPoint_params)
        else:
            for landmarkPoint in landmark_type.landmark:
                landmarkPoint_params = ([landmarkPoint.x, 
                                         landmarkPoint.y, 
                                         landmarkPoint.z])
                mpKpLandmarkArr.append(landmarkPoint_params)        
    else:
       #  If landmark is not available fill it with black array(black frame)
        mpKpLandmarkArr = np.zeros(int(points))
    return np.array(mpKpLandmarkArr).flatten()

# Concat all the landmark keypoints into a single array
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

def getFrameDetails(captureObject):
    msg = []
    frame_width  = captureObject.get(cv.CAP_PROP_FRAME_WIDTH)   
    frame_height = captureObject.get(cv.CAP_PROP_FRAME_HEIGHT) 
    frame_contrast = captureObject.get(cv.CAP_PROP_CONTRAST)
    frame_brightness = captureObject.get(cv.CAP_PROP_BRIGHTNESS)
    frame_hue = captureObject.get(cv.CAP_PROP_HUE)
    frame_saturation = captureObject.get(cv.CAP_PROP_SATURATION)
    frame_ps = captureObject.get(cv.CAP_PROP_FPS)
    msg.append("Frame Width: {}".format(frame_width))
    msg.append("Frame Height: {}".format(frame_height))
    msg.append("Frame Contrast: {}".format(frame_contrast))
    msg.append("Frame Brightness: {}".format(frame_brightness))
    msg.append("Frame Hue: {}".format(frame_hue))
    msg.append("Frame Saturation: {}".format(frame_saturation))
    msg.append("Frame Rate(Per Second): {}".format(frame_ps))
    return msg

textFont = cv.FONT_HERSHEY_SIMPLEX
textThickness = 1                   # Use integer value
textFontSize = 0.5

def setFrameDetails(frame, textFont, textFontSize, fontColor, textThickness, captureObj):
    label = "dummy data"
    width, height = cv.getTextSize(label, textFont, textFontSize, textThickness)[0]
    posHeight = 20 + height + 10
    frameDetails = getFrameDetails(captureObj)
    for line in frameDetails:
        print(line)
        cv.putText(frame, line, (20, int(posHeight)), textFont, textFontSize, fontColor, textThickness)
        posHeight = posHeight + height + 10
    
# Set mediapipe model
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    blackFrame = 255 * np.zeros(shape=(480, 320, 3), dtype=np.uint8)
    setFrameDetails(blackFrame, textFont, textFontSize, (100, 200, 100), textThickness, capObj)
    for detectItem in DETECTION_LIST:
        for videoCount in range(VIDEO_COUNT):
            for frameCount in range(FRAME_COUNT):      
                isTrue, frame = capObj.read()
                if not isTrue:
                    rootLogger.info("Application could not read frames from webcam. Exiting..")
                    capObj.release()
                    sys.exit()
                mpFrame, proessedFrame = mpProcessFrame(frame, holistic)
                drawLandmarksCustom(mpFrame, proessedFrame)
                
                finalFrame = np.concatenate((mpFrame, blackFrame), axis=1)
                # print(printFrameDetails(blackFrame, capObj))

                if frameCount == 0: 
                    cv.putText(
                        finalFrame, 
                        'STARTING COLLECTION', 
                        (600,400), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0,255, 0), 
                        1, 
                        cv.LINE_AA
                    )
                    cv.putText(
                        finalFrame, 
                        'Saving data for action, {}'.format(detectItem), 
                        (610,420), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 255), 
                        1, 
                        cv.LINE_AA
                    )
                    cv.imshow('OpenCV Feed', finalFrame)
                    cv.waitKey(5000)
                else: 
                    displayMsg = "Saving frames of {} video {}".format(detectItem, videoCount+1)
                    textWidth, textHeight = cv.getTextSize(displayMsg, textFont, textFontSize, textThickness)[0]
                    cv.rectangle(finalFrame, (600, 400), (590 + textWidth + 30, 410 + textHeight + 10), (255,255,255), -1)
                    cv.putText(
                        finalFrame, 
                        displayMsg,
                        (610,420), 
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5, 
                        (0, 255, 0), 
                        1,
                        cv.LINE_AA)
                    cv.imshow('OpenCV Feed', finalFrame)
                
                mpDetectionKeypoints = concatLandmarks(proessedFrame)
                # print(concatLandmarks(proessedFrame).shape) Result: (1662,)
                mpDetectionPoints = concatLandmarks(proessedFrame)
                savePath = os.path.join(DATA_PATH, detectItem, str(videoCount), str(frameCount))

                try:
                    np.save(savePath, mpDetectionPoints)
                    rootLogger.info("Sample data saved at {}".format(savePath))
                except:
                    rootLogger.exception("Unable to save sample data at {}".format(savePath))
                
                waitkey = cv.waitKey(10)
                if waitkey == 27:
                    rootLogger.warning("User pressed 'Esc' key. Exiting application...")
                    capObj.release()
                    sys.exit()

videoArr, labelArr = [], []
for action in DETECTION_LIST:
    for vdCount in range(VIDEO_COUNT):
        frameList = []
        for frame_num in range(FRAME_COUNT):
            frameFileName = os.path.join(DATA_PATH, action, str(vdCount), "{}.npy".format(frame_num))
            try:
                loadResponse = np.load(frameFileName)
                frameList.append(loadResponse)
                videoArr.append(frameList)
                labelArr.append(label_map[action])
            except:
                rootLogger.error("Unable to load sample file, {}".format(frameFileName))

X = np.array(videoArr)
Y = tf.keras.utils.to_categorical(labelArr)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(DETECTION_LIST.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, Y_train, epochs=500, callbacks=[tb_callback])

model.save('asl_prediction_model.h5')

# Delete model
# del model

# Rebuild the model again
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'))
# model.add(tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dense(DETECTION_LIST.shape[0], activation='softmax'))
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X_train, Y_train, epochs=500, callbacks=[tb_callback])

# Reload the model
# model.load_weights('asl_prediction_model.h5')

yhat = model.predict(X_test)
ytrue = np.argmax(Y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)

capObj.release()
cv.destroyAllWindows()
