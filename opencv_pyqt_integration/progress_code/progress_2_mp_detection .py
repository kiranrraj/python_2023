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
    # Default mode of frame received from opencv is in BGR
    # So we convert it into RGB format for mediapipe to 
    # detect the landmarks from the frame
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
    # The flags is used to reduce memeory utilization
    frameRGB.flags.writeable = False  
    # Processing the frame  
    processedFrame = model.process(frameRGB)            
    frameRGB.flags.writeable = True     
    # Converting image back into BGR format 
    frameBGR = cv.cvtColor(frameRGB, cv.COLOR_RGB2BGR) 
    return frameBGR, processedFrame

# Set mediapipe model
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capObj.isOpened():
        isTrue, frame = capObj.read()

        if not isTrue:
            print("Application could not read frames from webcam. Exiting..")
            capObj.release()
            sys.exit()

        mpFrame, proessedFrame = mpProcessFrame(frame, holistic)
        cv.imshow('OpenCV Feed', mpFrame)

        # print(proessedFrame)
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>
        # <class 'mediapipe.python.solution_base.SolutionOutputs'>

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

capObj.release()
cv.destroyAllWindows()
