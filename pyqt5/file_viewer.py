import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTreeView, QFileSystemModel, QVBoxLayout
from PyQt5.QtCore import QModelIndex

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

if __name__ == "__main__":
    app = QApplication([])
    win = FileSystemView()
    win.show()
    sys.exit(app.exec_())
