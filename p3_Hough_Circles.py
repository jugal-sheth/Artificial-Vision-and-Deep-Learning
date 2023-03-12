import cv2
import numpy as np

# Read image.
img = cv2.imread('Hough_Images/circle.jpg', cv2.IMREAD_COLOR)
cv2.imshow("Input", img)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))

# perform Hough transform to detect circles
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=100)

# If the circles are detected
if circles is not None:
    circles = np.uint16(np.around(circles))

# loop through list of circle to show all circles
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (0, 0, 255), 3)
        cv2.imshow("detected circles", img)
        cv2.waitKey(20)
    cv2.waitKey(0)

