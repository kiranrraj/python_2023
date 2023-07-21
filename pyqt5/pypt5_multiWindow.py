import sys

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTreeView, 
    QFileSystemModel
)

class FileSystemView(QWidget):
    def __init__(self):
        super().__init__()

        dir_path =r'D:\OpenCV\Project_try_1\Data'
        self.setWindowTitle("File Viewer")
        self.setFixedWidth(640)
        self.setFixedHeight(480)

        self.model = QFileSystemModel()
        self.model.setRootPath(dir_path)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(dir_path))

        winLayout = QVBoxLayout()
        winLayout.addWidget(self.tree)

        self.setLayout(winLayout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.newWindow = FileSystemView()

        winLayout = QVBoxLayout()
        winBtn = QPushButton("Open File Explorer")
        winBtn.clicked.connect(self.toggle_newWindow)
        winLayout.addWidget(winBtn)

        w = QWidget()
        w.setLayout(winLayout)
        self.setCentralWidget(w)

    def toggle_newWindow(self, checked):
        if self.newWindow.isVisible():
            self.newWindow.hide()

        else:
            self.newWindow.show()


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()