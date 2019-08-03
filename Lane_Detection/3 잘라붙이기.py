import numpy as np
import cv2

img = cv2.imread('C:/Users/images/land.jpg')
cv2.imshow('original', img)

subimg = img[300:400, 350:750]
cv2.imshow('cutting', subimg)

img[300:400, 0:400] = subimg

print(img.shape)
print(subimg.shape)

cv2.imshow('modified', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
