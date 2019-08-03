import numpy as np
import cv2

def onMouse(x):
    pass

def imgBlending():
    imgfile1 = 'C:/photo/images/land.jpg'
    imgfile2 = 'C:\photo\images/suzy.jpg'
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)
    img1 = img1[0:300, 0:250]
    img2 = img2[0:300, ]

    cv2.namedWindow("ImgPane")
    cv2.createTrackbar('MIXING', 'ImgPane', 0, 100, onMouse)
    mix = cv2.getTrackbarPos('MIXING', 'ImgPane')

    while True:
        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
        cv2.imshow('ImgPane', img)

        k=cv2.waitKey(1)
        if k==27:
            break

        mix=cv2.getTrackbarPos('MIXING', 'ImgPane')

    cv2.destroyAllWindows()

imgBlending()
