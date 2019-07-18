import numpy as np
import cv2

# BLUE : 120, 255, 255
lower_blue = np.array([110, 100, 100])
upper_blue = np.array([130, 255, 255])
# GREEN : 60, 255, 255
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])
# RED : 0, 255, 255
lower_red = np.array([-10, 100, 100])
upper_red = np.array([10, 255, 255])
# YELLOW : 21 110 220
custom_lower_yellow = np.array([11, 80, 100])
custom_upper_yellow = np.array([31, 255, 255])
# C_BLUE : 100 177 238
custom_lower_blue = np.array([90, 100, 100])
custom_upper_blue = np.array([110, 255, 255])


def channelplus(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, custom_lower_blue, custom_upper_blue)
    mask_yellow = cv2.inRange(hsv, custom_lower_yellow, custom_upper_yellow)

    mask_blue=cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)
    mask_yellow=cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)

    img = cv2.add(img, mask_blue)
    img = cv2.add(img, mask_yellow)
    return img
