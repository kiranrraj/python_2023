# Working // Checked
import os, time, sys, logging
import cv2 as cv                    
import numpy as np                  
import mediapipe as mp

mpHolistic = mp.solutions.holistic 
mpDrawing = mp.solutions.drawing_utils
capObj = cv.VideoCapture(1)

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

# Set mediapipe model
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capObj.isOpened():
        isTrue, frame = capObj.read()
        if not isTrue:
            print("Application could not read frames from webcam. Exiting..")
            capObj.release()
            sys.exit()
        mpFrame, proessedFrame = mpProcessFrame(frame, holistic)
        drawLandmarksCustom(mpFrame, proessedFrame)
        cv.imshow('OpenCV Feed', mpFrame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

capObj.release()
cv.destroyAllWindows()
