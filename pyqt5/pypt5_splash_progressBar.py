import time, sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, 
    QWidget, 
    QProgressBar, 
    QLabel, QFrame, 
    QVBoxLayout
    )


class MySplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(480, 140)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: green;")

        self.progress = 0
        self.step = 200

        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.progressFunction)
        self.timer.start(30)

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.frame = QFrame()
        layout.addWidget(self.frame)

        self.labelTitle = QLabel(self.frame)
        self.labelTitle.setObjectName('LabelTitle')

        self.myPBar = QProgressBar(self.frame)
        self.myPBar.setAlignment(Qt.AlignCenter)
        self.myPBar.setFormat('%p%')
        self.myPBar.setTextVisible(True)
        self.myPBar.setRange(0, self.step)
        self.myPBar.setGeometry(100, 20, 280, 20)

        self.myLabel = QLabel(self.frame)
        self.myLabel.setAlignment(Qt.AlignCenter)
        self.myLabel.setText('Loading')
        self.myLabel.setGeometry(200, 40, 80, 60)
        self.myLabel.setStyleSheet("QLabel"
                                 "{"
                                 "font-size : 20px;"
                                 "color : yellow;"
                                 "}")

    def progressFunction(self):
        self.myPBar.setValue(self.progress)

        if self.progress >= self.step:
            self.timer.stop()
            self.close()

            time.sleep(1)

            self.myApp = MyApp()
            self.myApp.show()

        self.progress += 1

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    
    app = QApplication([])

    splash = MySplashScreen()
    splash.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')