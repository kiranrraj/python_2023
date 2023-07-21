# Working // Checked
import os, time, sys, logging
import cv2 as cv                    
import numpy as np                  
import mediapipe as mp

mpHolistic = mp.solutions.holistic 
mpDrawing = mp.solutions.drawing_utils
capObj = cv.VideoCapture(1)

# Mediapipe detection function
def mpProcessFrame(frame, model):
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
    frameRGB.flags.writeable = False    
    processedFrame = model.process(frameRGB)            
    frameRGB.flags.writeable = True     
    frameBGR = cv.cvtColor(frameRGB, cv.COLOR_RGB2BGR) 
    return frameBGR, processedFrame

# Custom drawing for landmarks. The function take output of the mediapipe processed image and result
def drawLandmarksCustom(frame, processedFrame):
    mpDrawing.draw_landmarks(frame, 
                             processedFrame.left_hand_landmarks, 
                             mpHolistic.HAND_CONNECTIONS, 
                             mpDrawing.DrawingSpec(         # Dot Properties
                                    color=(121,22,76), 
                                    thickness=1, 
                                    circle_radius=1), 
                             mpDrawing.DrawingSpec(         # Line Properties
                                    color=(121,44,250),
                                    thickness=2,
                                    circle_radius=1)
                             )  
    mpDrawing.draw_landmarks(frame, 
                             processedFrame.right_hand_landmarks, 
                             mpHolistic.HAND_CONNECTIONS, 
                             mpDrawing.DrawingSpec(
                                    color=(245,117,66), 
                                    thickness=2, 
                                    circle_radius=4), 
                             mpDrawing.DrawingSpec(
                                    color=(245,66,230), 
                                    thickness=2, 
                                    circle_radius=2)
                             ) 
    mpDrawing.draw_landmarks(frame, processedFrame.face_landmarks, mpHolistic.FACEMESH_TESSELATION, 
                             mpDrawing.DrawingSpec(
                                    color=(80,110,10), 
                                    thickness=1, 
                                    circle_radius=1), 
                             mpDrawing.DrawingSpec(
                                    color=(80,256,121), 
                                    thickness=1, 
                                    circle_radius=1)
                             ) 
    mpDrawing.draw_landmarks(frame, 
                             processedFrame.pose_landmarks, 
                             mpHolistic.POSE_CONNECTIONS,
                             mpDrawing.DrawingSpec(
                                    color=(80,22,10), 
                                    thickness=2, 
                                    circle_radius=4), 
                             mpDrawing.DrawingSpec(
                                    color=(80,44,121), 
                                    thickness=2, 
                                    circle_radius=2)
                             ) 
# Get landmark key points and put it into a array
def getMPLandmarks(mpFrame, mpLandmark, points):
    mpKpLandmarkArr = []
    landmark_type = getattr(mpFrame, mpLandmark)
    if landmark_type != None:
        if mpLandmark == "pose_landmarks":
       #      Only pose landmark have the visibility property
            for landmarkPoint in landmark_type.landmark:
                landmarkPoint_params = ([landmarkPoint.x, 
                                         landmarkPoint.y, 
                                         landmarkPoint.z, 
                                         landmarkPoint.visibility])
                mpKpLandmarkArr.append(landmarkPoint_params)
        else:
            for landmarkPoint in landmark_type.landmark:
                landmarkPoint_params = ([landmarkPoint.x, 
                                         landmarkPoint.y, 
                                         landmarkPoint.z])
                mpKpLandmarkArr.append(landmarkPoint_params)        
    else:
       #  If landmark is not available fill it with black array(black frame)
        mpKpLandmarkArr = np.zeros(int(points))
    return np.array(mpKpLandmarkArr).flatten()

# Concat all the landmark keypoints into a single array
def concatLandmarks(mp_out):
    pose_coordinates = getMPLandmarks(mp_out, "pose_landmarks", 132)
    face_coordinates = getMPLandmarks(mp_out, "left_hand_landmarks", 63)
    left_hand_coordinates = getMPLandmarks(mp_out, "right_hand_landmarks", 63)
    right_hand_coordinates = getMPLandmarks(mp_out, "face_landmarks", 1404)
    return np.concatenate([
        pose_coordinates, 
        face_coordinates, 
        left_hand_coordinates, 
        right_hand_coordinates
    ])

# Set mediapipe model
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capObj.isOpened():
        isTrue, frame = capObj.read()
        if not isTrue:
            print("Application could not read frames from webcam. Exiting..")
            capObj.release()
            sys.exit()
        mpFrame, proessedFrame = mpProcessFrame(frame, holistic)
        drawLandmarksCustom(mpFrame, proessedFrame)
        cv.imshow('OpenCV Feed', mpFrame)
        mpDetectionKeypoints = concatLandmarks(proessedFrame)
       #  print(concatLandmarks(proessedFrame).shape) Result: (1662,)
        waitkey = cv.waitKey(10)
        if waitkey == 27:
            break



# for kp in proessedFrame.pose_landmarks.landmark:
#     kp_arr = np.array([kp.x, kp.y, kp.z, kp.visibility])
#     print(kp_arr)

# [ 0.4863674   0.79358    -0.98180163  0.99354184]
# [ 0.48518854  0.73597831 -0.94781482  0.99264485]
# [ 0.49211907  0.7338919  -0.94770783  0.99287671]
# [ 0.49954313  0.73169744 -0.94763649  0.99267292]
# [ 0.45674482  0.74420291 -0.99987519  0.99283195]
# [ 0.44585523  0.75342369 -1.00015569  0.99231446]
# [ 0.43252021  0.75483966 -1.00033808  0.99113268]
# [ 0.4977037   0.75218028 -0.64148504  0.99312615]
# [ 0.39553183  0.78593719 -0.86938179  0.99032068]
# [ 0.50009674  0.83493942 -0.8427639   0.97216678]
# [ 0.46549386  0.8471489  -0.91399497  0.97157568]
# [ 0.5332545   0.94393533 -0.39971566  0.88085854]
# [ 0.30760485  1.02271998 -0.63772404  0.71217692]
# [ 0.61691052  1.11073554 -0.23075536  0.09868016]
# [ 0.34293246  1.20467031 -0.72591406  0.07173397]
# [ 0.60813522  1.0253166  -0.22589211  0.20865749]
# [ 0.4122673   1.06434476 -0.83449489  0.08284399]
# [ 0.60859174  1.0059725  -0.22639266  0.26810744]
# [ 0.42914423  1.03873718 -0.86616182  0.12082188]
# [ 0.60435522  0.98043525 -0.23556587  0.31593701]
# [ 0.42008829  1.01595962 -0.83801061  0.15059018]
# [ 0.5980168   0.99042809 -0.23094639  0.29241785]
# [ 0.41644502  1.025069   -0.83045685  0.13870256]
# [0.55976504 1.60736036 0.01288113 0.00320571]
# [ 0.3914434   1.63740182 -0.01088321  0.00241132]
# [ 0.57278502  1.82977438 -0.01412221  0.00820823]
# [0.41186905 1.86008203 0.07222689 0.00348363]
# [6.27674639e-01 2.17532706e+00 4.08682585e-01 7.10039225e-04]
# [4.80123699e-01 2.15831757e+00 4.94480371e-01 5.47163247e-04]
# [6.38324857e-01 2.23880911e+00 4.37760830e-01 5.69201657e-04]
# [4.88679469e-01 2.20145488e+00 5.30235112e-01 5.41545043e-04]
# [0.62788022 2.25908375 0.17691958 0.00245897]
# [5.01310408e-01 2.25625944e+00 2.98963696e-01 2.06065853e-03]



capObj.release()
cv.destroyAllWindows()
