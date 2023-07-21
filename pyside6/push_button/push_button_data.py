import sys
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("My Window")
        self.setFixedSize(QSize(480, 360))
        self.myBtn = QPushButton("Click")
        self.myBtn.setCheckable(True)
        self.setCentralWidget(self.myBtn)
        self.myBtn.clicked.connect(self.btnClicked)

    def btnClicked(self, checked):
        print("Button Clicked", checked) 

if __name__ == "__main__":
    app = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec()

