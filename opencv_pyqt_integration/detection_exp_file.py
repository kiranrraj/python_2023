import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import sys, os, time
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QGridLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt

SAMPLE_SIZE = 100
DATA_DIR = r"D:\github\python_2023\opencv_pyqt_integration\image"

class WorkerThread(QThread):
    imageUpdateSignal = pyqtSignal(QImage)
    mp_holistics = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    def run(self):
        self.threadStatus = True,
        capObj = cv.VideoCapture(0)
        with self.mp_holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
            while self.threadStatus:
                isTrue, currentFrame = capObj.read()
                
                if isTrue:
                    processed_img, mp_model = self.mp_detection(currentFrame, holistic_model)
                    currentFrameRgb = cv.cvtColor(processed_img, cv.COLOR_BGR2RGB)
                    
                        # print(mp_model)
                    self.draw_landmarks(currentFrameRgb, mp_model)
                    currentFrameRgbFlipped = cv.flip(currentFrameRgb, 1)
                    frameToQtFormat = QImage(currentFrameRgbFlipped.data, 
                                            currentFrameRgbFlipped.shape[1], 
                                            currentFrameRgbFlipped.shape[0],
                                            QImage.Format_RGB888)
                    scaledFrame = frameToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.imageUpdateSignal.emit(scaledFrame)

    def stop(self):
        self.threadStatus = False
        self.quit()

    def draw_landmarks(self, img, landmark_list):
        # FACE_CONNECTIONS seems to be renamed/replaced by FACEMESH_TESSELATION.
        self.mp_drawing.draw_landmarks(img, landmark_list.face_landmarks, 
                                self.mp_holistics.FACEMESH_TESSELATION, 
                                self.mp_drawing.DrawingSpec(color=(65,255,0), thickness=1, circle_radius=1), 
                                self.mp_drawing.DrawingSpec(color=(0, 59, 0), thickness=1, circle_radius=1))
        
        self.mp_drawing.draw_landmarks(img, landmark_list.pose_landmarks, 
                                self.mp_holistics.POSE_CONNECTIONS, 
                                self.mp_drawing.DrawingSpec(color=(65,255,0), thickness=1, circle_radius=1), 
                                self.mp_drawing.DrawingSpec(color=(0, 59, 0), thickness=1, circle_radius=1))
        
        self.mp_drawing.draw_landmarks(img, landmark_list.left_hand_landmarks, 
                                self.mp_holistics.HAND_CONNECTIONS, 
                                self.mp_drawing.DrawingSpec(color=(65,255,0), thickness=1, circle_radius=1), 
                                self.mp_drawing.DrawingSpec(color=(0, 59, 0), thickness=1, circle_radius=1))
        
        self.mp_drawing.draw_landmarks(img, landmark_list.right_hand_landmarks, 
                                self.mp_holistics.HAND_CONNECTIONS, 
                                self.mp_drawing.DrawingSpec(color=(65,255,0), thickness=1, circle_radius=1), 
                                self.mp_drawing.DrawingSpec(color=(0, 59, 0), thickness=1, circle_radius=1))

    def mp_detection(self, frame, model):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        mp_output = model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        return frame_bgr, mp_output

    def showSeperate(self):
        capObj = cv.VideoCapture(0)
        with self.mp_holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
            while capObj.isOpened():
                isTrue, video_frame = capObj.read()

                if isTrue:
                    processed_img, mp_model = self.mp_detection(video_frame, holistic_model)
                    # print(mp_model)
                    self.draw_landmarks(processed_img, mp_model)
                    cv.imshow("frame", processed_img)

                    key = cv.waitKey(20)
                    if key == ord('q'):
                        break
                sample = 0
                while sample < SAMPLE_SIZE:
                    isTrue, frame = capObj.read()
                    if isTrue:
                        # cv.imshow('frame', processed_img)
                        cv.waitKey(30)
                        cv.imwrite(os.path.join(DATA_DIR, '{}.jpg'.format(sample)), frame)
                    sample += 1
            capObj.release()
            cv.destroyAllWindows()

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.mainLayout = QGridLayout()
        self.frameLabel = QLabel()
        self.mainLayout.addWidget(self.frameLabel)
        self.startBtn = QPushButton("Start Feed")
        self.startBtn.clicked.connect(self.startFeed)
        self.mainLayout.addWidget(self.startBtn)
        self.stopBtn = QPushButton("Stop Feed")
        self.mainLayout.addWidget(self.stopBtn)
        self.stopBtn.clicked.connect(self.stopFeed)
        self.showBtn = QPushButton("Show Feed")
        self.mainLayout.addWidget(self.showBtn)
        self.showBtn.clicked.connect(self.showNewWindow)
        self.setLayout(self.mainLayout)
        self.workerFeed = WorkerThread()

    def startFeed(self):
        self.workerFeed.start()
        self.workerFeed.imageUpdateSignal.connect(self.imageSlot)

    def imageSlot(self, img):
        self.frameLabel.setPixmap(QPixmap.fromImage(img))

    def stopFeed(self):
        self.workerFeed.stop()

    def showNewWindow(self):
        self.workerFeed.showSeperate()

if __name__ == "__main__":
    App = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(App.exec_())