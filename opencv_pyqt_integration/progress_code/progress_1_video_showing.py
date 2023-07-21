# Working // Checked
import os, time, sys, logging
import cv2 as cv                    
import numpy as np                  
import mediapipe as mp

capObj = cv.VideoCapture(0)
while capObj.isOpened():
    isTrue, frame = capObj.read()

    if not isTrue:
        print("Application could not read frames from webcam. Exiting..")
        capObj.release()
        sys.exit()

    cv.imshow('OpenCV Feed', frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

capObj.release()
cv.destroyAllWindows()
