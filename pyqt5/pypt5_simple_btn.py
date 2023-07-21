from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtCore import QSize

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # change the title of our main window.
        self.setWindowTitle("My App")

        btn = QPushButton("Click Me", self)

        # Set the size of the button
        btn.resize(100,50)

        # Set the position of the button
        btn.move(320 - 50, 240 - 25)

        # Set the function to be called when the button is clicked
        btn.clicked.connect(self.clickMethod)

        # .setCentralWidget to place a widget in the QMainWindow 
        # by default it takes the whole of the window.
        # self.setCentralWidget(btn)

        # resize your applications
        # Qt sizes are defined using a QSize object. 
        # This accepts width and height parameters
        self.setFixedSize(QSize(640, 440))

    def clickMethod(self):
        print("Button Clicked")

app = QApplication([])
window = MainWindow()
window.show()
app.exec_()