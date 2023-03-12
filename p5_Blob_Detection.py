import cv2
import numpy as np

img = cv2.imread("Blob_images/Blob1.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("input", img)

# blob detector parameters (set to find squares)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = False
params.filterByConvexity = False
params.filterByInertia = False
params.filterByCircularity = True
params.minCircularity = 0.8
params.maxCircularity = 1

detector = cv2.SimpleBlobDetector_create(params)
blobs = detector.detect(img)
b = len(blobs)
for i in range(b):
    x, y = blobs[i].pt
    print("Center of the blob: ({}, {})".format(x, y))
# draw detected blobs to the colour image
img_blob = cv2.drawKeypoints(img, blobs, np.array([]), (0, 0, 255),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# show image
cv2.imshow("res", img_blob)
cv2.waitKey(0)
