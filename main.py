import os
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Variables
width, height = 1080, 720
folderPath = "Images"

# camera Setup
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of Images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# Variables
imageNumber = 0
hs, ws = int(180 * 1), int(270 * 1)
gestureThreshold = 400
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = 0
annotationStart = False

# Hand Detector
detector = HandDetector(detectionCon=0.5, maxHands=1)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imageNumber])
    imageCurrent = cv.imread(pathFullImage)

    hands, frame = detector.findHands(frame)
    horizontalLine = cv.line(frame, (gestureThreshold, gestureThreshold), (width, gestureThreshold),
                             (255, 0, 0), 10)
    verticalLine = cv.line(frame, (gestureThreshold, 0), (gestureThreshold, gestureThreshold),
                           (255, 0, 0), 10)
    print(annotationNumber)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width-200], [0, width-200]))
        yVal = int(np.interp(lmList[8][1], [150, height-200], [0, height-200]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:
            if cx >= gestureThreshold:
                # Gesture 1 - Left
                if fingers == [1, 0, 0, 0, 0]:
                    annotationStart = False
                    if imageNumber > 0:
                        buttonPressed = True
                        annotations = [[]]
                        annotationNumber = 0
                        imageNumber -= 1
                    else:
                        print("This is the first slide")

                # Gesture 2 - Right
                if fingers == [0, 0, 0, 0, 1]:
                    annotationStart = False
                    if imageNumber < len(pathImages) - 1:
                        buttonPressed = True
                        annotations = [[]]
                        annotationNumber = 0
                        imageNumber += 1
                    else:
                        print("This is the last slide")

        # Gesture 3 - Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv.circle(imageCurrent, indexFinger, 12, (0, 0, 255), cv.FILLED)

        # Gesture 4 - Draw Pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv.circle(imageCurrent, indexFinger, 12, (0, 0, 255), cv.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

        # Gesture 5 - Erase Annotations
        if fingers == [1, 1, 1, 1, 1]:
            if annotations:
                if annotationNumber >= 0:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True

    # Button Pressed iterations
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv.line(imageCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

    # Adding Webcam on Presentation
    imageSmall = cv.resize(frame, (ws, hs))
    h, w, _ = imageCurrent.shape
    imageCurrent[0:hs, w-ws:w] = imageSmall

    cv.imshow('frame', frame)
    cv.imshow('Presentation', imageCurrent)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
