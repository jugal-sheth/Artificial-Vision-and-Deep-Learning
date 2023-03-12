import cv2
import numpy as np

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([132, 255, 255])
lower_red = np.array([160, 80, 80])
upper_red = np.array([179, 255, 255])
lower_green = np.array([50, 100, 100])
upper_green = np.array([72, 255, 255])

font = cv2.FONT_HERSHEY_SIMPLEX
org = (460, 400)
fontScale = 1.5
color = (0, 255, 255)
thickness = 3

color_car = 0
for n in range(30):
    path = "Car_Images/" + str(n) + ".png"
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    blue_pixels = cv2.countNonZero(mask_blue)
    green_pixels = cv2.countNonZero(mask_green)
    red_pixels = cv2.countNonZero(mask_red)
    # print(blue_pixels, green_pixels, red_pixels)
    if blue_pixels > red_pixels + green_pixels:
        color_car = "BLUE"
    elif green_pixels > red_pixels + blue_pixels:
        color_car = "GREEN"
    elif red_pixels > green_pixels + blue_pixels:
        color_car = "RED"
    else:
        print("no pixel")
    print("Car Number " + str(n) + " is possibly " + color_car + " car")
    img = cv2.putText(img, color_car, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("image", img)
    cv2.waitKey(1000)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
