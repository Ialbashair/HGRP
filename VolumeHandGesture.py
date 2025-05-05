import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()  # min: -96 , max: 0
volBar = 400
vol = 0
volPercent = 0
minVol = volRange[0]
maxVol = volRange[1]

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevTime = 0

detector = htm.handDetector(detectionCon=0.8)

while True:
    success, img = cap.read()

    img = detector.FindHands(img, True)
    lmList = detector.FindPostion(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[8])
        # print(lmList[4])

        # grab positions of thumb and index tips
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx , cy = (x1+x2)// 2, (y1+ y2) // 2 # Center of line

        # draw line between thumb and index tips
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        length = math.hypot(x2-x1, y2-y1) # get length of line
        #print(length)

        # Hand Range 20 - 420
        # Volume Range -96 - 0


        vol = np.interp(length, [20, 420], [minVol, maxVol])
        volBar = np.interp(length, [20, 420], [400, 150])
        volPercent = np.interp(length, [20, 420], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

    # volume bar
    cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    # volume percent
    cv2.putText(img, f' {int(volPercent)}', (48, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)

    # Find and write FPS
    currentTime = time.time()
    fps = 1/(currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, f'FPS: {int(fps)}', (48,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)

    cv2.imshow('image', img)
    cv2.waitKey(1)
