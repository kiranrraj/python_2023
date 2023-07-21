import logging
import os
import sys

from PyQt5 import QtCore, QtWidgets


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(
    level=logging.DEBUG, format=" %(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class MyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Setup logging here:
        self.logTextBox = QTextEditLogger(self)
        self.logTextBox.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(self.logTextBox)
        logging.getLogger().setLevel(logging.DEBUG)

        self._button = QtWidgets.QPushButton()
        self._button.setText("Start")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.logTextBox.widget)
        layout.addWidget(self._button)

        self._button.clicked.connect(self.test)

        self.process = QtCore.QProcess()
        self.process.readyReadStandardOutput.connect(
            self.handle_readyReadStandardOutput
        )
        self.process.started.connect(lambda: print("Started!"))
        self.process.finished.connect(lambda: print("Finished!"))

    def test(self):
        logging.debug("damn, a bug")
        logging.info("something to remember")
        logging.warning("that's not right")
        logging.error("foobar")

        script = os.path.join(CURRENT_DIR, "another_module.py")
        self.process.start(sys.executable, [script])

    def handle_readyReadStandardOutput(self):
        text = self.process.readAllStandardOutput().data().decode()
        self.logTextBox.widget.appendPlainText(text.strip())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dlg = MyDialog()
    dlg.show()
    dlg.raise_()
    sys.exit(app.exec_())