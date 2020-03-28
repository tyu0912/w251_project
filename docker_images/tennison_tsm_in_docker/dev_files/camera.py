import cv2
import time

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 331)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 331)

while True:
    
    try:
        ret, img = cap.read()
        cv2.imshow('vis', img)

    except:
        pass

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

