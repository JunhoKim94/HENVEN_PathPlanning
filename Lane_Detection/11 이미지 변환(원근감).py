import numpy as np
import cv2

img = cv2.imread('C:\photo\images/blue_lane.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape[:2]
pts1 = np.float32([[50, 50], [200, 50], [20, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

#기울기변환 (구겨짐) 매트릭스
M = cv2.getAffineTransform(pts1, pts2)
img4 = cv2. warpAffine(img, M, (w,h))

cv2.imshow('original', img)
cv2.imshow('Affine-Transform', img4)

pts1 = np.float32([[0,0], [300, 0], [0, 300], [300, 300]])
pts2 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

#원근감 매트릭스
M2 = cv2.getPerspectiveTransform(pts1, pts2)

img5 = cv2.warpPerspective(img, M2, (w, h))
cv2.imshow('Perspective-Transform', img5)


cv2.waitKey(0)
cv2.destroyAllWindows()
