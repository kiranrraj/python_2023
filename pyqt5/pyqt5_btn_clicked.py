from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(0, 0, 640, 480)
        self.setWindowTitle("My Window")
        self.initUI()
    
    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText("My label")

        self.btn = QtWidgets.QPushButton(self)
        self.btn.setText("Click")
        self.btn.move(280, 240)
        self.btn.clicked.connect(self.btn_click)

    def btn_click(self):
        self.label.setText("Button clicked")

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()

