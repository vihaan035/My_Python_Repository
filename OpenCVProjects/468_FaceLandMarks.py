import time
import cv2
import cvzone
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
cap = cv2.VideoCapture(1)
cap.set(3,468)
cap.set(4, 648)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
pTime = 0
segmentor = SelfiSegmentation()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70),cv2.FONT_HERSHEY_COMPLEX,
                2,(0,255,0),1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
