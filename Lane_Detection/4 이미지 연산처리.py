import numpy as np
import cv2

def addImage():
    imgfile1 = 'C:/photo/images/land.jpg'
    imgfile2 = 'C:\photo\images/suzy.jpg'
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    img1 = img1[0:300, 0:250]
    img2 = img2[0:300, ]

    cv2.imshow('img1', img1)
    cv2.imshow('igm2', img2)

    add_img1 = img1 + img2
    add_img2 = cv2.add(img1, img2)

    cv2.imshow('img1+img2', add_img1)
    cv2.imshow('add(img1, img2)', add_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

addImage()
