'''
    cv2.adaptveThreshold(img, value,
    adaptiveMethod, thresholdType, blocksize, c)

    img: Grayscale 이미지
    value: adaptiveMethod에 의해 계산된 문턱값과 thresholdType에 의해
    픽셀에 적용 될 최대값
    adaptiveMethod: 사용할 Adaptive Thresholding 알고리즘
    
    cv2.ADAPTIVE_THRES_MEAN_C: 적용할 픽셀 (x,y)를 중심으로 하는 blocksize
    *blocksize 안에 있는 픽셀 값의 평균에서 C를 뺀 값을 문턱값으로 함
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 적용할 픽셀 (x,y)를 중심으로 하는 blocksize*
    blocksize안에 있는 Gaussian 윈도우 기반 가중치들의 합에서 C를 뺀 값을 문턱
    값으로 함

    blocksize: 픽셀에 적용할 문턱값을 계산하기 위한 블럭 크기, 적용될 픽셀이
    블럭의 중심이 됨. 따라서 blocksize는 홀수여야 함
    C: 보정 상수로, 이 값이 양수이면 계산된 adaptive 문턱값에서 빼고, 음수면
    더해줌, 0이면 그대로...
'''

import numpy as np
import cv2

img = cv2.imread('C:\photo\images/blue_lane.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
thr2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thr3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#Otsu 바이너리제이션
ret, thr4 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#가우시안 블러 적용 후 Otsu 바이너리제이션
blur = cv2.GaussianBlur(img, (5,5), 0)
ret, thr5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#bilateral 필터 적용
blur2 = cv2.bilateralFilter(img, 6, 50, 50)
ret, thr6 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

titles = ['original', 'Global Thresholding(v=127)', 'Adaptive MEAN', 'Adaptive GAUSSIAN'
          , 'Otsu Binary', 'Gaussian + Otsu', 'bilateral+Otsu']
images = [img, thr1, thr2, thr3, thr4, thr5, thr6]

for i in range(7):
    cv2.imshow(titles[i], images[i])

cv2.waitKey(0)
cv2.destroyAllWindows()

