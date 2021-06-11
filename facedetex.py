import cv2
import time

import imutils

import faceDetection as fd

cap = cv2.VideoCapture(0)
pTime = 0
detector = fd.FaceDetector()
while True:
    success, frame = cap.read()
    img = imutils.resize(frame, width=720)
    img, bboxs = detector.findFaces(img)
    print(bboxs)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
