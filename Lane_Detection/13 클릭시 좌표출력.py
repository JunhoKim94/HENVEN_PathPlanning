
import numpy as np
import cv2

def print_cordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(img[y, x])
        print(x, y)

img = cv2.imread('C:\photo\images\images_shor.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
print(img.shape[:2])
cv2.imshow("img", img)
cv2.setMouseCallback("img", print_cordinate)
