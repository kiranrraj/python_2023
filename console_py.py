from PyQt5 import QtCore, QtGui, QtWidgets


class LogView(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._process = QtCore.QProcess()
        self._process.readyReadStandardOutput.connect(self.handle_stdout)
        self._process.readyReadStandardError.connect(self.handle_stderr)

    def start_log(self, program, arguments=None):
        if arguments is None:
            arguments = []
        self._process.start(program, arguments)

    def add_log(self, message):
        self.appendPlainText(message.rstrip())

    def handle_stdout(self):
        message = self._process.readAllStandardOutput().data().decode()
        self.add_log(message)

    def handle_stderr(self):
        message = self._process.readAllStandardError().data().decode()
        self.add_log(message)


if __name__ == "__main__":

    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = LogView()
    w.resize(640, 480)
    w.show()
    w.start_log("adb", ["logcat", "*:I"])

    sys.exit(app.exec_())