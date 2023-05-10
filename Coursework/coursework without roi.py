import cv2
import numpy as np
import time

# Path to fetch the video
cap = cv2.VideoCapture("Coursework_Videos/video1.mp4")

# parameters for the blob detection algorithm
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 400
params.maxArea = 1400
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

# Record the start time
start_time = time.time()

# blob detector with the specified parameters
detector = cv2.SimpleBlobDetector_create(params)

# initialize variables for counting the blobs
number_of_circles_crossed = 0
number_of_lines_crossed = 0

# main loop
while True:
    # read each frame as a image
    success, img = cap.read()
    if not success:
        # Break main loop if video has ended
        print("Video has been ended")
        break
    # control the speed of video with cv wait key similar to delay function
    # 0 is infinite delay where new frame is read onl after a keystroke
    # 1 to any number decides delay in milliseconds
    cv2.waitKey(10)
    cv2.imshow("Input", img)

    # Only specific area ou of the entire video is selected with boundary box coordinates to speed up processing
    # ROI = Region of Interest
    roi = img
    # convert the ROI in HSV colourspace
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # kernel dimensions for erode and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Dilate The HSV of ROI
    hsv = cv2.dilate(hsv, kernel, iterations=1)

    # Erode The HSV of ROI
    hsv = cv2.erode(hsv, kernel, iterations=1)
    cv2.imshow("HSV", hsv)

    # Create Mask To separate RED blobs from video frame ROI
    mask = cv2.inRange(hsv, (0, 0, 0), (30, 255, 225))
    imask = mask > 0
    red = np.zeros_like(roi, np.uint8)
    red[imask] = roi[imask]

    # Display Red Blobs
    cv2.imshow("Blobs", red)

    # Create Grayscale ROI for finding blobs
    gray_roi_blob = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Blob", gray_roi_blob)

    # Create Grayscale ROI for finding outer wraps
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray ", gray_roi)

    # Blur the Gray image to detect circles
    gray_blurred = cv2.blur(gray_roi, (3, 3))

    cv2.imshow("Gray Blurred", gray_blurred)
    # Threshold Gray image
    ret, thresh = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY)

    # Display the Threshold image
    cv2.imshow("Threshold", thresh)

    # Detect Circles Using Hough Circles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10,maxRadius=30)

    # Create Keypoint for each Blob
    keypoints = detector.detect(gray_roi_blob)

    # loop through the detected blobs
    for keypoint in keypoints:
        # draw a circle around the blob and label it
        # Draw a filled Circle with RED colour for blob
        cv2.circle(img, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size / 2), (0, 0, 255), -1)

        # Draw center of blob with BLACK colour
        cv2.circle(img, (int(keypoint.pt[0]), int(keypoint.pt[1])), 3, (0, 0, 0), -1)

        # Put Text near each detected Blob
        cv2.putText(img, "Blob", (int(keypoint.pt[0]) - 25, int(keypoint.pt[1]) + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw circles around Outer sheath
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:

            # coordinates of center of circle for outer sheath and add Y1 parameter to y axis
            center = (i[0], i[1])

            # Draw a WHITE center at center of outer sheath
            cv2.circle(img, center, 1, (255, 255, 255), 2)
            radius = i[2]

            # Draw a circle around outer sheath with WHITE colour
            cv2.circle(img, center, radius, (255, 255, 255), 1)

            # Put tech near each detected Wrap
            cv2.putText(img, "Wrap", (int(center[0]) - 25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)

            # script to count the blobs
            if 280 > center[0] > 220:
                if number_of_lines_crossed == 0:
                    number_of_lines_crossed = 1

            elif 220 > center[0] > 190:
                if number_of_lines_crossed == 1:
                    number_of_lines_crossed = 2

            elif center[0] < 190:
                if number_of_lines_crossed == 2:
                    number_of_circles_crossed += 1
                    number_of_lines_crossed = 0

    # convert number of circles crossed variable to string to display on image
    count = str(number_of_circles_crossed)

    # lines for counting Blobs
    cv2.line(img, (280, 0), (280, 160), (255, 0, 0), 1)
    cv2.line(img, (220, 0), (220, 160), (0, 0, 255), 1)
    cv2.line(img, (190, 0), (190, 160), (0, 255, 0), 1)

    cv2.putText(img, "Total Bubbles Crossed line =", (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, count, (930, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # display the final output
    cv2.imshow("Output Detected Sheaths and Blobs", img)

end_time = time.time()  # Record the end time
print("Total time: ", end_time - start_time)