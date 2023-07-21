from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QLabel("Click in this window")
        self.setCentralWidget(self.label)

    def mouseMoveEvent(self, e):
        self.label.setText("mouseMoveEvent")

    def mousePressEvent(self, e):
        print("--------------------------------------------------------------")
        if e.button() == Qt.LeftButton:
            self.label.setText("Mouse Click Event: Left Button")

        elif e.button() == Qt.MiddleButton:
            self.label.setText("Mouse Click Event: Middle Button")

        elif e.button() == Qt.RightButton:
            self.label.setText("Mouse Click Event: Right Button")

        print(e.buttons())
        print(e.position())
        print("X position relative to the widget: {}".format(e.position().x()))
        print("Y position relative to the widget: {}".format(e.position().y()))
        print("--------------------------------------------------------------")
        self.label.setText("mousePressEvent")

    def mouseReleaseEvent(self, e):
        print("--------------------------------------------------------------")
        if e.button() == Qt.LeftButton:
            self.label.setText("Mouse Release Event: Left Button")

        elif e.button() == Qt.MiddleButton:
            self.label.setText("Mouse Release Event: Middle Button")

        elif e.button() == Qt.RightButton:
            self.label.setText("Mouse Release Event: Right Button")

        print(e.buttons())
        print(e.position())
        print("X position relative to the widget: {}".format(e.position().x()))
        print("Y position relative to the widget: {}".format(e.position().y()))
        print("--------------------------------------------------------------")
        self.label.setText("mouseReleaseEvent")

    def mouseDoubleClickEvent(self, e):
        print("--------------------------------------------------------------")
        if e.button() == Qt.LeftButton:
            self.label.setText("Mouse Double Click Event: Left Button")

        elif e.button() == Qt.MiddleButton:
            self.label.setText("Mouse Double Click Event: Middle Button")

        elif e.button() == Qt.RightButton:
            self.label.setText("Mouse Double Click Event: Right Button")

        print(e.buttons())
        print(e.position())
        print("X position relative to the widget: {}".format(e.position().x()))
        print("Y position relative to the widget: {}".format(e.position().y()))
        print("--------------------------------------------------------------")
        self.label.setText("mouseDoubleClickEvent")

app = QApplication([])
window = MainWindow()
window.show()
app.exec()