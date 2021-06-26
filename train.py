import cv2
import numpy as np
import time
import poseEstimationModule as pm

cap = cv2.VideoCapture('ai/exercise.mp4')
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success,img = cap.read()
    img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("AiTrainer/test.jpg")
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
