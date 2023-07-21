####################### Import Section #####################
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * #QDesktopWidget, QStatusBar
from PyQt5.QtCore import *
from datetime import datetime
from sklearn.model_selection import train_test_split as train
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import sys, time, webbrowser, logging, os, webbrowser, subprocess, urllib, threading
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import tensorflow as tf
##########################################################

################## Constants #############################
command_string = "tensorboard --logdir=summaries"
tensorBoard_uri = r'http://localhost:6006/'
log_location = r'D:\OpenCV\Project_try_1\asl_detection.log'
dir_path =r'D:\OpenCV\Project_try_1\Data'
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
current_path = r"D:\OpenCV\Project_try_1"
data_location = os.path.join(current_path, "Data")

text_1 = "Starting Collection"
text_font = cv.FONT_HERSHEY_SIMPLEX
text_font_size = 1
text_font_size_sec = 0.5
text_font_thickness = 3
text_font_thickness_sec = 1
text_color = (255, 0, 150)
text_color_sec = (255, 255, 150)
(text_width, text_height), baseline = cv.getTextSize(text_1, text_font, text_font_size, text_font_thickness)

sequence_number = 30
sequence_length = 30
actions = np.array(['Hello', 'Thanks', 'ILoveYou'])
label_map = {label:item for item, label in enumerate(actions)}

mpHolistics = mp.solutions.holistic
mpDrawing = mp.solutions.drawing_utils
##########################################################


##############  Logging Setting  ###################### 
logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("{}\\asl_detection.log".format(current_path))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
#######################################################


############## Splash Screen ##########################
class MySplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(480, 140)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: green;")
        self.progress = 0
        self.step = 2
        self.initUI()
        self.timer = QTimer()
        self.timer.timeout.connect(self.progressFunction)
        self.timer.start(30)

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        center(self)
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
            self.myApp = MyWindow1()
            self.myApp.show()
        self.progress += 1
#####################################################

####### Center the window on the middle of the screen ########
def center(parent):
    frame_geo = parent.frameGeometry()
    desk_geo = QDesktopWidget().availableGeometry().center()
    frame_geo.moveCenter(desk_geo)
    parent.move(frame_geo.topLeft())
##############################################################

############ Check Tensor Board is available #################
def check_tensorBoard(uri):
    site_up = False
    while not site_up:
        try:
            conn = urllib.request.urlopen(uri)
        except:
            rootLogger.error("Error.. Waiting 10 seconds")
            time.sleep(10)
        else:
            site_up = True
            print(f'{uri} is up')
    return True
###############################################################

####### Application Sstart, Stop, Elapsed time Details ########
def print_start_time():
    start_time = datetime.now()
    rootLogger.info("Application started at {}".format(start_time))
    rootLogger.info("Application started at {}".format(start_time))
    return start_time

def print_stop_time():
    stop_time = datetime.now()
    rootLogger.info("Application started at {}".format(stop_time))
    return stop_time

def print_elapsed_time(start, stop):
    time_diff = (stop - start).total_seconds() * 1000000
    rootLogger.info("Application ran for {} seconds".format(int(time_diff)))

def get_frame_details(video_obj):
    width = int(video_obj.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_obj.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video_obj.get(cv.CAP_PROP_FPS)
    return width, height, fps
####################################################################

# def display_frame_details(bg_image):
#     cv.putText(bg_image, "Video Resolution {} x {}.".format(video_width, video_height), (10, 80), 
#             text_font, text_font_size_sec, text_color_sec, text_font_thickness_sec)
#     cv.putText(bg_image, "Video Frame Rate {}.".format(video_fps), (10, 110), 
#             text_font, text_font_size_sec, text_color_sec, text_font_thickness_sec)

# def display_frame_count(bg_image, item_type, frame_num):
#     cv.putText(bg_image, "Collecting frames for {}.".format(item_type), (10, 20), 
#             text_font, text_font_size_sec, text_color_sec, text_font_thickness_sec)
#     cv.putText(bg_image, "Video frame number {}.".format(frame_num), (10, 50), 
#             text_font, text_font_size_sec, text_color_sec, text_font_thickness_sec)

################## Sample Folder Creation ############################
def create_sample_folders(data_directory):
    for item in actions:
        for number in range(sequence_number):
            folder_url = os.path.join(data_directory, item, str(number))
            rootLogger.info("Folder {} created.".format(folder_url))
            try:
                os.makedirs(folder_url)
            except:
                rootLogger.warning("Folder not created".format(folder_url))
####################################################################

##################### Sample File Explorer #########################
class FileSystemView(QWidget):
    def __init__(self):
        super().__init__()
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
####################################################################

class MyWindow1(QWidget):
    def __init__(self):
        super(MyWindow1, self).__init__()
        self.newWindow = FileSystemView()
        self.setWindowTitle("QGridLayout")
        center(self)
        self.i = 0

        # Create a QGridLayout instance
        self.winLayout = QGridLayout()

        # Add widgets to the layout
        self.frameFeed = QLabel()
        self.frameFeed.setFixedSize(480, 360)
        self.winLayout.addWidget(self.frameFeed, 0, 0, 2, 5)
        self.videoDetails = QTextBrowser()
        self.videoDetails.setDisabled(True)
        self.winLayout.addWidget(self.videoDetails,0, 5, 2, 7)

        # Push Buttons
        self.videoBtn = QPushButton("Video Feed")
        self.winLayout.addWidget(self.videoBtn, 2, 0)
        self.videoBtn.clicked.connect(self.start_feed)
        self.trainBtn = QPushButton("Training")
        self.winLayout.addWidget(self.trainBtn, 2, 1)
        self.trainBtn.clicked.connect(self.startSampling)
        self.detectionBtn = QPushButton("ASL Detection")
        self.winLayout.addWidget(self.detectionBtn, 2, 2)
        self.tensorBBtn = QPushButton("Tensor Board")
        self.winLayout.addWidget(self.tensorBBtn,2, 3)
        self.tensorBBtn.clicked.connect(self.createTensorBThread)
        self.cancelBtn = QPushButton("Cancel")
        self.winLayout.addWidget(self.cancelBtn,2, 4)
        self.cancelBtn.clicked.connect(self.stop_feed)
        self.fileExpBtn = QPushButton("Open Samples")
        self.winLayout.addWidget(self.fileExpBtn,2, 5)
        self.fileExpBtn.clicked.connect(self.toggleNewWindow)
        self.loadLogBtn = QPushButton("Load Log")
        self.winLayout.addWidget(self.loadLogBtn,2, 6)
        self.loadLogBtn.clicked.connect(self.displayLog)

        self.logDisplay = QTextBrowser()
        self.winLayout.addWidget(self.logDisplay,3, 0, 2, 5)

        self.statusBar = QStatusBar()
        self.winLayout.addWidget(self.statusBar,7, 0, 1, 7)
        self.statusBar.showMessage("Application")
        self.statusBar.setStyleSheet("background-color : pink")
        
        self.thread = QThread()
        self.worker2 = WorkerFunction()
        self.worker2.moveToThread(self.thread)
        self.thread.started.connect(self.worker2.run)

        self.worker2.imgUpdate.connect(self.imgSlot)
        self.worker2.finished.connect(self.worker_done)
        self.thread.finished.connect(self.thread_done)

        self.setLayout(self.winLayout)

    def getWindowsDetails(self, frame):
        print("Get details")

    def imgSlot(self, img):
        self.frameFeed.setPixmap(QPixmap.fromImage(img))

    def start_feed(self):
        self.worker2.camera = cv.VideoCapture(0)
        self.worker2.running = True
        self.worker2.detection = False
        self.thread.start()
        self.get_details()

    def get_details(self):
        frame = self.worker2.camera
        frame_width  = frame.get(cv.CAP_PROP_FRAME_WIDTH)   
        frame_height = frame.get(cv.CAP_PROP_FRAME_HEIGHT) 
        frame_cotrast = frame.get(cv.CAP_PROP_CONTRAST)
        frame_brightness = frame.get(cv.CAP_PROP_BRIGHTNESS)
        frame_hue = frame.get(cv.CAP_PROP_HUE)
        frame_saturation = frame.get(cv.CAP_PROP_SATURATION)
        frame_ps = frame.get(cv.CAP_PROP_FPS)
###########################################################
        msg = """Video Width: {}
Video Height: {}
FPS: {}
Contrast: {}
Brightness: {}
Hue: {}
Saturation: {}""".format(
            frame_width, 
            frame_height, 
            frame_ps, 
            frame_cotrast, 
            frame_brightness, 
            frame_hue, 
            frame_saturation
            )
        self.videoDetails.setPlainText(msg)
###########################################################
    def stop_feed(self):
        self.worker2.camera.release()
        print("feed was asked to stop")

    def worker_done(self):
        print("worker finished")
        self.worker2.camera.release()
        self.thread.quit()

    def thread_done(self):
        print("thread finished")

    def toggleNewWindow(self):
        if self.newWindow.isVisible():
            self.newWindow.hide()
        else:
            self.newWindow.show()

    def openTensorBoard(self):
        try:
            self.statusBar.showMessage("Pleaase wait... Accessing Tensor Board...")
            sub_process_out = subprocess.Popen(command_string)
            if check_tensorBoard(tensorBoard_uri):
                webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
                webbrowser.get('chrome').open(tensorBoard_uri)
                self.statusBar.showMessage("Opening Tensor Board on {}".format(tensorBoard_uri))
        except subprocess.CalledProcessError as e:
            rootLogger.error("Error !!!")
            self.statusBar.showMessage("Unable to open Tensor Board")

    def createTensorBThread(self):
        newTrainingThread = threading.Thread(target=self.openTensorBoard)
        newTrainingThread.start()

    def startSampling(self):
        create_sample_folders(data_location)
        self.worker2.camera = cv.VideoCapture(0)
        self.worker2.running = True
        self.worker2.detection = True
        self.thread.start()
        self.get_details()

    def displayLog(self):
        self.logDisplay.clear()
        if os.path.isfile(log_location):
            print("File found")
            try: 
                text=open(log_location).read()
                self.logDisplay.setPlainText(text)
            except:
                rootLogger.error("Error Occurred!!!! Unable to read log.")
        else:
            self.statusBar.showMessage("File not found")
            self.logDisplay.setPlainText("Log Not Found in {}".format(log_location))
            rootLogger.error("Log Not Found in {}".format(log_location))
                    
    def checkFileFolders(self):
        sample_collection_flag = True
        for item in actions:
            for number in range(sequence_number):
                folder = os.path.join(data_location, item, str(number))
                if not os.path.exists(folder):
                    rootLogger.error("Folder: {} not found.".format(folder))
                    sample_collection_flag = False
                else:
                    for frame_num in range(sequence_length):
                        file = os.path.join(data_location, item, str(number),"{}.npy".format(frame_num))
                        if not os.path.exists(file):
                            rootLogger.error("File: {} not found.".format(file))
                            sample_collection_flag = False
        return sample_collection_flag

def filesCheck(location, items, numbers, framesNum):
    files = 0
    for item in items:
        for number in range(numbers):
            for frameNum in range(framesNum):
                fileName = os.path.join(location, item, str(number),"{}.npy".format(frameNum))
                if os.path.exists(fileName):
                    files +=1
    if files == len(items) * numbers * framesNum:
        return False
    else:
        return True

def mpDetection(img, model):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgRGB.flags.writeable = False

    imgOut = model.process(imgRGB)

    imgRGB.flags.writeable = True
    imgBGR = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)

    return imgBGR, imgOut

# Function to draw points and connections
def draw_landmarks(img, imgOut):
    # FACE_CONNECTIONS seems to be renamed/replaced by FACEMESH_TESSELATION.
    mpDrawing.draw_landmarks(img, imgOut.face_landmarks, 
        mpHolistics.FACEMESH_TESSELATION, 
        mpDrawing.DrawingSpec(color=(65,255,0), thickness=1, circle_radius=1), 
        mpDrawing.DrawingSpec(color=(0, 59, 0), thickness=1, circle_radius=1))
    
    mpDrawing.draw_landmarks(img, imgOut.pose_landmarks, 
        mpHolistics.POSE_CONNECTIONS, 
        mpDrawing.DrawingSpec(color=(65,255,0), thickness=1, circle_radius=1), 
        mpDrawing.DrawingSpec(color=(0, 59, 0), thickness=1, circle_radius=1))
    
    mpDrawing.draw_landmarks(img, imgOut.left_hand_landmarks, 
        mpHolistics.HAND_CONNECTIONS, 
        mpDrawing.DrawingSpec(color=(65,255,0), thickness=1, circle_radius=1), 
        mpDrawing.DrawingSpec(color=(0, 59, 0), thickness=1, circle_radius=1))
    
    mpDrawing.draw_landmarks(img, imgOut.right_hand_landmarks, 
        mpHolistics.HAND_CONNECTIONS, 
        mpDrawing.DrawingSpec(color=(65,255,0), thickness=1, circle_radius=1), 
        mpDrawing.DrawingSpec(color=(0, 59, 0), thickness=1, circle_radius=1))

def get_mp_landmarks(mp_out, mp_landmark, points):
    mp_array = []
    landmark_type = getattr(mp_out, mp_landmark)
    if landmark_type != None:
        if mp_landmark == "pose_landmarks":
            for lm_point in landmark_type.landmark:
                lm_point_params = ([lm_point.x, lm_point.y, lm_point.z, lm_point.visibility])
                mp_array.append(lm_point_params)
        else:
            for lm_point in landmark_type.landmark:
                lm_point_params = ([lm_point.x, lm_point.y, lm_point.z])
                mp_array.append(lm_point_params)        
    else:
        mp_array = np.zeros(int(points))
    return np.array(mp_array).flatten()

def concat_landmarks(mp_out):
    pose_coordinates = get_mp_landmarks(mp_out, "pose_landmarks", 132)
    face_coordinates = get_mp_landmarks(mp_out, "left_hand_landmarks", 63)
    left_hand_coordinates = get_mp_landmarks(mp_out, "right_hand_landmarks", 63)
    right_hand_coordinates = get_mp_landmarks(mp_out, "face_landmarks", 1404)
    return np.concatenate([pose_coordinates, face_coordinates, left_hand_coordinates, right_hand_coordinates])

def savingSamples(location, items, numbers, framesNum, frame):
    i = 0
    flag = filesCheck(data_location, actions, sequence_number, sequence_length)
    if flag:
        for item in items:
            for number in range(numbers):
                for frameNum in range(framesNum):
                    dataLocation = os.path.join(location, item, str(number), str(frameNum))
                    i+=1
                    print("{}.Saving for {}. Sequence:{} Frame Number:{} ".format(i, item, str(number), str(frameNum)))
                    print("Press s to save the frame")
                    
                    np.save(dataLocation, frame)
                    print("{}. Files saved {}".format(i, dataLocation))


def image_detection(imgFrame):
        with mpHolistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            img, imgOut = mpDetection(imgFrame, holistic)
            draw_landmarks(img, imgOut)
            mp_keypoints = concat_landmarks(imgOut)
            return img
        

class WorkerFunction(QThread):
    finished = pyqtSignal()
    imgUpdate = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.camera = None
        self.running = None
        self.detection = None

    def image_convertion(self, frame):
        rgbframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        flipFrame = cv.flip(rgbframe, 1)
        cQT_frame = QImage(flipFrame.data, flipFrame.shape[1],
                            flipFrame.shape[0], QImage.Format.Format_RGB888)
        img = cQT_frame.scaled(640, 480, Qt.KeepAspectRatio)
        return img
    
    def np_to_pyQtImg(self, img):
        height, width, channel = img.shape
        bytesPerLine = channel * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def run(self):
        while self.running:
            isTrue, frame = self.camera.read()

            if isTrue:
                if self.detection:
                    img = image_detection(frame)
                    out_img = self.np_to_pyQtImg(img)
                else:
                    out_img = self.image_convertion(frame)
                self.imgUpdate.emit(out_img)
            else:
                logging.error("Unable to read from the webcam. Exiting...")
                exit("Unable to read from the webcam. Exiting...")

        print("\nfinished signal emited")
        self.finished.emit()
    











if __name__ == "__main__":
    myApp = QApplication([])
    splash = MySplashScreen()
    splash.show()

    try:
        sys.exit(myApp.exec_())
    except SystemExit:
        rootLogger.info('Closing Window...')