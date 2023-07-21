# Working // Checked
import os, time, sys, logging
import cv2 as cv                    
import numpy as np                  
import mediapipe as mp

mpHolistic = mp.solutions.holistic 
mpDrawing = mp.solutions.drawing_utils
capObj = cv.VideoCapture(0)

# Mediapipe detection function
def mpProcessFrame(frame, model):
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
    frameRGB.flags.writeable = False    
    processedFrame = model.process(frameRGB)            
    frameRGB.flags.writeable = True     
    frameBGR = cv.cvtColor(frameRGB, cv.COLOR_RGB2BGR) 
    return frameBGR, processedFrame

# Draw landmark using default values
def drawLandmarksDefault(frame, processedFrame):
    mpDrawing.draw_landmarks(frame, processedFrame.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS) 
    mpDrawing.draw_landmarks(frame, processedFrame.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
    mpDrawing.draw_landmarks(frame, processedFrame.face_landmarks, mpHolistic.FACEMESH_TESSELATION) 
    mpDrawing.draw_landmarks(frame, processedFrame.pose_landmarks, mpHolistic.POSE_CONNECTIONS)

# Set mediapipe model
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capObj.isOpened():
        isTrue, frame = capObj.read()
        if not isTrue:
            print("Application could not read frames from webcam. Exiting..")
            capObj.release()
            sys.exit()
        mpFrame, proessedFrame = mpProcessFrame(frame, holistic)
        drawLandmarksDefault(mpFrame, proessedFrame)
        cv.imshow('OpenCV Feed', mpFrame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

capObj.release()
cv.destroyAllWindows()
