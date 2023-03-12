import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("Coursework_Videos/Video1.mp4")


def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high
    H_low = cv2.getTrackbarPos('low H', 'controls')
    H_high = cv2.getTrackbarPos('high H', 'controls')
    S_low = cv2.getTrackbarPos('low S', 'controls')
    S_high = cv2.getTrackbarPos('high S', 'controls')
    V_low = cv2.getTrackbarPos('low V', 'controls')
    V_high = cv2.getTrackbarPos('high V', 'controls')


cv2.namedWindow('controls')
H_low = 0
H_high = 179
S_low = 0
S_high = 255
V_low = 0
V_high = 255

cv2.createTrackbar('low H', 'controls', 0, 179, callback)
cv2.createTrackbar('high H', 'controls', 179, 179, callback)

cv2.createTrackbar('low S', 'controls', 0, 255, callback)
cv2.createTrackbar('high S', 'controls', 255, 255, callback)

cv2.createTrackbar('low V', 'controls', 0, 255, callback)
cv2.createTrackbar('high V', 'controls', 255, 255, callback)

while True:
    success, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("video", img)
    hsv_low = np.array([H_low, S_low, V_low])
    hsv_high = np.array([H_high, S_high, V_high])
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(img, img, mask=mask)
    #cv2.waitKey(1000)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # cv2.imshow('mask', mask)
    cv2.imshow('Result', final)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
