import os, sys
import cv2 as cv
import logging

ALPHABETS = 3
SAMPLE_SIZE = 100
ROOT_DIR = r'D:\github\python_2023\project_experiments'
DATA_DIR = os.path.join(ROOT_DIR, "Training_data")
LOG_DIR = os.path.join(ROOT_DIR, "Logs")
print(DATA_DIR)

ALPHABETS_TO_TRAIN = {0:"a", 1:"b", 2:"c"}

def create_data_dir(uri):
    if not os.path.exists(uri):
        os.makedirs(uri)
        rootLogger.info('Directory created. {}'.format(uri))

##############  Logging Setting  ###################### 
logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
create_data_dir(LOG_DIR)
fileHandler = logging.FileHandler("{}\\asl_detection.log".format(LOG_DIR))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
#######################################################

create_data_dir(ROOT_DIR)
capObj = cv.VideoCapture(0)

for alphabet in range(ALPHABETS):
    folder_uri = os.path.join(DATA_DIR, str(alphabet))
    create_data_dir(folder_uri)

    rootLogger.info('Collecting data for class {}'.format(alphabet))

    flag = False
    while capObj.isOpened():
        isTrue, frame = capObj.read()
        if isTrue:
            cv.putText(frame, "Press 'r' when you are ready", (100, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,cv.LINE_AA)
            cv.imshow('Video Feed', frame)
            if cv.waitKey(30) == ord('r'):
                break
    sample = 0
    while sample < SAMPLE_SIZE:
        isTrue, frame = capObj.read()
        if isTrue:
            cv.imshow('Video Feed', frame)
            cv.waitKey(30)
            cv.imwrite(os.path.join(DATA_DIR, str(alphabet), '{}.jpg'.format(sample)), frame)
        sample += 1
capObj.release()
cv.destroyAllWindows()