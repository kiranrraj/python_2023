import sys, os, logging
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QPlainTextEdit, QGridLayout
from PyQt5.QtCore import QProcess

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(
    level=logging.DEBUG, format=" %(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Command Executor")
        self.resize(480, 360)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(40)

        self.commandWindow = QPlainTextEdit()
        self.grid.addWidget(self.commandWindow, 0, 0, 2, 2)

        self.resultWindow = QPlainTextEdit()
        self.grid.addWidget(self.resultWindow, 2, 0, 2, 2)

        self.runBtn = QPushButton("Execute")
        self.grid.addWidget(self.runBtn, 4, 0)
        self.runBtn.clicked.connect(self.getCommand)

        self.clearBtn = QPushButton("Clear")
        self.grid.addWidget(self.clearBtn, 4, 1)
        self.clearBtn.clicked.connect(self.clearFields)
        
        self.setLayout(self.grid)

    def getCommand(self):
        userCommand = self.commandWindow.toPlainText().strip()
        if userCommand:
            try:
                output = os.popen(userCommand)
            except:
                self.resultWindow.insertPlainText("Error!!!!")

            if output:
                self.commandWindow.clear()
                outputText = output.read()
                self.resultWindow.insertPlainText(outputText)          

    def clearFields(self):
        self.commandWindow.clear()
        self.resultWindow.clear()

if __name__ == "__main__":
    app = QApplication([])
    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec_())
    except:
        print("Closing window")


