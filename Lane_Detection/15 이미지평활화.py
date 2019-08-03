import numpy as np
import cv2

def smoothing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)
    return img2

video="C:\photo\images/challenge.avi"
cap = cv2.VideoCapture(video)
while True:
    ret, img = cap.read()
    if not ret:
        print('비디오 끝')
        break
    cv2.waitKey(33)
    cv2.imshow('ori', img)
    sm_img=smoothing(img)
    cv2.imshow('smt', sm_img)
