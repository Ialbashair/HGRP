import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevTime = 0

detector = htm.handDetector(detectionCon=0.8)

def set_mute(mute_status):
    # Get default audio device
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Mute (1) or Unmute (0)
    volume.SetMute(mute_status, None)

while True:
    success, img = cap.read()

    img = detector.FindHands(img, True)
    lmList = detector.FindPostion(img, False)
    strippedList = [sublist[1:] for sublist in lmList]

    fingerTips = [4,8,12,16,20]
    lineConnection = [
        (4, 8),  # 1 → 3
        (8, 12),  # 1 → 2
        (12, 16),  # 2 → 5
        (16, 20),  # 5 → 3
        (20, 4),  # 3 → 4
    ]

    if len(lmList) != 0:
        # Draw Line
        for (i, j) in lineConnection:
             cv2.line(img, strippedList[i], strippedList[j], (255, 0, 0), 2)  # Blue line

        fingertips = [strippedList[i] for i in fingerTips]  # List of (x, y) tuples

        # Find centroid
        # Calculate centroid (ONCE per frame)
        centroid_x = sum(pt[0] for pt in fingertips) / len(fingerTips)
        centroid_y = sum(pt[1] for pt in fingertips) / len(fingerTips)
        centroid = (int(centroid_x), int(centroid_y))

        # find distance between center point and fingertips
        distances = []
        for fingertip in fingertips:
            dist = math.hypot(fingertip[0] - centroid[0], fingertip[1] - centroid[1])
            distances.append(dist)
        # Draw centroid (ONCE per frame)
        cv2.circle(img, centroid, 5, (0, 0, 255), -1)  # Red filled circle

        if sum(distances) > 150:
            set_mute(0)
        if sum(distances) < 150:
            set_mute(1)

            # Find and write FPS
    currentTime = time.time()
    fps = 1/(currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, f'FPS: {int(fps)}', (48, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)

    cv2.imshow('image', img)
    cv2.waitKey(1)
