from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import sys

class MainWindow(QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__()
		self.available_cameras = QCameraInfo.availableCameras()
		self.viewfinder = QCameraViewfinder()
		self.viewfinder.show()
		self.setCentralWidget(self.viewfinder)
		self.select_camera(0)
		self.setWindowTitle("PyQt5 Cam")
		self.show()

	def select_camera(self, i):
		self.camera = QCamera(self.available_cameras[i])
		self.camera.setViewfinder(self.viewfinder)
		self.camera.start()

if __name__ == "__main__" :
    App = QApplication([])
    window = MainWindow()
    sys.exit(App.exec())
