import sys
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QGridLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("My Window")
        self.setFixedSize(QSize(480, 360))
        self.myLayout = QGridLayout()
        self.myBtn = QPushButton("B1 [Click Me]")
        self.myBtn.setCheckable(True)
        self.myLayout.addWidget(self.myBtn)
        self.myBtn.clicked.connect(self.btnClicked)
        self.myBtn1 = QPushButton("Reset")
        self.myLayout.addWidget(self.myBtn1)
        self.myBtn1.clicked.connect(self.btnReset)

        widget = QWidget()
        widget.setLayout(self.myLayout)
        self.setCentralWidget(widget)

    def btnClicked(self, checked):
        self.myBtn.setText("You Already Clicked")
        self.myBtn.setEnabled(False)
        self.setWindowTitle("Button Disabled")
        print("Button Clicked", checked) 

    def btnReset(self):
        print(self.myBtn.isEnabled())
        if not self.myBtn.isEnabled():
            self.myBtn.setText("B1 [Click Me]")
            self.myBtn.setEnabled(True)
            self.setWindowTitle("My Window")

if __name__ == "__main__":
    app = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec()

