from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

xpos = 0
ypos = 0
width = 640
height = 480

def window(x, y, w, h):
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(x, y, w, h)
    win.setWindowTitle("My Window")

    btn = QtWidgets.QPushButton(win)
    btn.setText("Click")
    btn.move(int(w/2), int(h/2))

    win.show()
    sys.exit(app.exec_())

window(xpos, ypos, width, height)
