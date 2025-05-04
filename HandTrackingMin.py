import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # create hands object and pass default parameters for now
mpDraw = mp.solutions.drawing_utils # draw utils

prevTime = 0
currTime = 0


while True:
    success, img = cap.read() # read capture
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert our image to rgb to feed to hands object
    results = hands.process(imgRGB)  # process frame and give result
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks: # check for hand
        for handLms in results.multi_hand_landmarks: # for each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape # find width and height of img
                cx, cy = int(lm.x * w), int(lm.y * h) # multiply landmark x and y values by height and width of img
                if id == 4:
                    cv2.circle(img, (cx, cy), 25, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draw hand points and connections

    # fps counter
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    # write fps to screen
    cv2.putText(img, str(int(fps)), (5,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)


