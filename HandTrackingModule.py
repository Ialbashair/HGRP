import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False, maxHands = 2, modelComp = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionCon, self.trackCon)  # create hands object and pass default parameters for now
        self.mpDraw = mp.solutions.drawing_utils  # draw utils

    # Converts image to RGB
    def ConvertImg(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert our image to rgb to feed to hands object
        self.results = self.hands.process(imgRGB)  # process frame and give result
        return self.results

    def FindHands(self, img, draw=True):
        result = self.ConvertImg(img)
        if result.multi_hand_landmarks:  # check for hand
            for handLms in result.multi_hand_landmarks:  # for each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # draw hand points and connections
        return img

    def FindPostion(self, img, handNum = 0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # find width and height of img
                cx, cy = int(lm.x * w), int(lm.y * h)  # multiply landmark x and y values by height and width of img
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)

    prevTime = 0
    currTime = 0

    detector = handDetector()
    while True:
        success, img = cap.read()  # read capture

        img = detector.FindHands(img,True) # call findHands method to get img
        lmList = detector.FindPostion(img, draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        # fps counter
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # write fps to screen
        cv2.putText(img, str(int(fps)), (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
