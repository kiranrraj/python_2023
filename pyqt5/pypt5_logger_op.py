import sys
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QTextCursor
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QTextEdit, QPushButton, QVBoxLayout

class WriteStream(object):
    def __init__(self,queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

class MyReceiver(QObject):
    mysignal = pyqtSignal(str)

    def __init__(self,queue):
        QObject.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal.emit(text)

# An example QObject (to be run in a QThread) which outputs information with print
class LongRunningThing(QObject):
    def run(self):
        for i in range(1000):
            print (i)

# An Example application QWidget containing the textedit to redirect stdout to
class MyApp(QtWidgets.QWidget):
    def __init__(self,*args,**kwargs):
        QWidget.__init__(self,*args,**kwargs)

        self.layout = QVBoxLayout(self)
        self.textedit = QTextEdit()
        self.button = QPushButton('start long running thread')
        self.button.clicked.connect(self.start_thread)
        self.layout.addWidget(self.textedit)
        self.layout.addWidget(self.button)

    def append_text(self,text):
        self.textedit.moveCursor(QTextCursor.End)
        self.textedit.insertPlainText( text )

    def start_thread(self):
        self.thread = QThread()
        self.long_running_thing = LongRunningThing()
        self.long_running_thing.moveToThread(self.thread)
        self.thread.started.connect(self.long_running_thing.run)
        self.thread.start()

# Create Queue and redirect sys.stdout to this queue
queue = Queue()
sys.stdout = WriteStream(queue)

# Create QApplication and QWidget
qapp = QApplication([])  
app = MyApp()
app.show()

# Create thread that will listen on the other end of the queue, and send the text to the textedit in our application
thread = QThread()
my_receiver = MyReceiver(queue)
my_receiver.mysignal.connect(app.append_text)
my_receiver.moveToThread(thread)
thread.started.connect(my_receiver.run)
thread.start()

qapp.exec_()