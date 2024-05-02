import cv2
import numpy as np
import HandTrackingModule as htm

headerList = ["blue.png", "red.png", "yellow.png", "eraser.png", "save.png"]

overlayList = []
for i in headerList:
    image = cv2.imread(f'toolbox/{i}')
    overlayList.append(image)

# default header, color, canva size
canvas = np.zeros((720, 1280, 3), np.uint8)
header = overlayList[0]
color = (255, 110, 0)
detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0 

# camera setting
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while 1:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    nodes = detector.findPosition(img, draw=False)
 
    if len(nodes) != 0:

        # 2nd, 3rd fingers' landmarks
        x1, y1 = nodes[8][1:]
        x2, y2 = nodes[12][1:]
 
        fingers = detector.fingersUp()
        
        # drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
 
            cv2.line(img, (xp, yp), (x1, y1), color, 25)
 
            thickness = 100 if color == (0, 0, 0) else 25
            cv2.line(img, (xp, yp), (x1, y1), color, thickness)
            cv2.line(canvas, (xp, yp), (x1, y1), color, thickness)
 
            # 不加會畫成像孔雀開屏一樣
            xp, yp = x1, y1
 
        # selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            
            # fingers are in header area
            if y1 < 125:
                if x1 < 200: # save the painting
                    header = overlayList[4]
                    cv2.imwrite("your-painting.png", canvas)
                if x1 > 1050:
                    header = overlayList[3]
                    color = (0, 0, 0)
                elif x1 > 800:
                    header = overlayList[2]
                    color = (0, 220, 255)
                elif x1 > 550:
                    header = overlayList[1]
                    color = (50, 50, 255)
                elif x1 > 250:
                    header = overlayList[0]
                    color = (255, 110, 0)
 
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # extract the paint
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    
    # update the paint on camera img
    img = cv2.bitwise_and(img, inv)
    
    # add up the paint color from canvas and camera img
    img = cv2.bitwise_or(img, canvas)
 
    
    # Setting the header image
    img[0:125, 0:1280] = header
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Image", img)
    cv2.waitKey(1)