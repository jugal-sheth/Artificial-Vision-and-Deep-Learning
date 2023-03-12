import cv2
import numpy as np
import math as m

thr = 100
minLength = 100
maxGap = 100
img = cv2.imread("Hough_Images/Hough3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.blur(gray, (3, 3))
edge = cv2.Canny(img, 50, 200, None, 3)
cv2.imshow("img", img)
cv2.waitKey(1000)
coloured_edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
coloured_edge_p = np.copy(coloured_edge)
lines = cv2.HoughLines(edge, 1, np.pi / 180, thr, None, minLength, maxGap)
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = m.cos(theta)
        b = m.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(coloured_edge, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("res", coloured_edge)

linesP = cv2.HoughLinesP(edge, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        linP = linesP[i][0]
        cv2.line(coloured_edge_p, (linP[0], linP[1]), (linP[2], linP[3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("prob", coloured_edge_p)
    cv2.waitKey(0)
