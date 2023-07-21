# Integrating OpenCV into PyQt5

# Imports
import cv2 as cv
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QWidget, QApplication
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QPixmap, QImage
import sys

class MyWindow1(QWidget):
    def __init__(self):
        super(MyWindow1, self).__init__()

        # Creating a layout for the window
        self.winLayout = QVBoxLayout()

        self.frameFeed = QLabel()
        self.winLayout.addWidget(self.frameFeed)

        self.trainBtn = QPushButton("Train")
        self.winLayout.addWidget(self.trainBtn)

        self.detectionBtn = QPushButton("ASL Detection")
        self.winLayout.addWidget(self.detectionBtn)

        self.cancelBtn = QPushButton("Cancel")
        self.winLayout.addWidget(self.cancelBtn)
        self.cancelBtn.clicked.connect(self.exitFeed)

        self.Worker1 = WorkerFunction()
        self.Worker1.start()
        self.Worker1.imgUpdate.connect(self.imgSlot)
        self.setLayout(self.winLayout)

    def imgSlot(self, img):
        self.frameFeed.setPixmap(QPixmap.fromImage(img))

    def exitFeed(self):
        self.Worker1.stop()

class WorkerFunction(QThread):
    imgUpdate = pyqtSignal(QImage)
    def run(self):
        self.tActive = True
        capObj = cv.VideoCapture(0)
        while capObj.isOpened() and self.tActive:
            isTrue, frame = capObj.read()

            if isTrue:
                rgbframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                flipFrame = cv.flip(rgbframe, 1)
                cQT_frame = QImage(flipFrame.data, flipFrame.shape[1],
                                   flipFrame.shape[0], QImage.Format.Format_RGB888)
                img = cQT_frame.scaled(640, 480, Qt.KeepAspectRatio)
                self.imgUpdate.emit(img)
            
        capObj.release()
    
    def stop(self):
        self.tActive = False
        self.quit()

if __name__ == "__main__":
    MyApp = QApplication(sys.argv)
    MyWindow = MyWindow1()
    MyWindow.show()
    sys.exit(MyApp.exec_()) 


