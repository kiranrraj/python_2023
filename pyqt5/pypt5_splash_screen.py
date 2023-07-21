from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtCore import Qt, QTimer
import sys

app = QApplication([])
myLabel = QLabel("<Font color=Red size=16>This is a splash Screen</font>")
myLabel.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
myLabel.setFixedWidth(300)
myLabel.setFixedHeight(100)
myLabel.setAlignment(Qt.AlignCenter)
myLabel.show()

QTimer.singleShot(2000, app.quit)

sys.exit(app.exec_())