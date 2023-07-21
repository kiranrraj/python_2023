from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        layout = QVBoxLayout()

        self.label = QLabel()
        layout.addWidget(self.label)

        self.textField = QLineEdit()
        layout.addWidget(self.textField)
        self.textField.textChanged.connect(self.label.setText)

        myWidget = QWidget()
        myWidget.setLayout(layout)
        self.setCentralWidget(myWidget)


app = QApplication([])
window = MainWindow()
window.show()
app.exec()