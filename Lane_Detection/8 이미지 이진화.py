import numpy as np
import cv2

img = cv2.imread('C:\photo\images/image_straight.jpg', cv2.IMREAD_GRAYSCALE)

'''
    cv2.threshold(img, threshold_value, value, flag)
    img : Garyscale 이미지
    value : 픽셀 문턱값보다 클 때 적용되는 최대값 혹은 그 반대
    flag: 문턱값 적용 방법 또는 스타일

    cv2.THRESH_BINARY: 픽셀 값이 threshold_value 보다 크면 value 작으면 0
    cv2.THRESH_BINARY_INV: 픽셀 값이 threshold_value 보다 크면 0 작으면 value
    cv2.THRESH_TRUNC: 픽셀 값이 threshold_value 보다 크면
    threshold_value 작으면 0
    cv2.THRESH_TOZERO: 픽셀 값이 threshold_value 보다 크면 픽셀 값 그대로 작으면 0
    cv2.THRESH_TOZERO_INV: 픽셀 값이 threshold_value 보다 크면 0 작으면 0 픽셀
    값 그대로
'''

ret, thr1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thr2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thr3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thr4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thr5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow('ol', ret)
cv2.imshow('original', img)
cv2.imshow('BINARY', thr1)
cv2.imshow('BINARY_INV', thr2)
cv2.imshow('TRUNC', thr3)
cv2.imshow('TOZERO', thr4)
cv2.imshow('TOZERO_INV', thr5)

cv2.waitKey(0)
cv2.destroyAllWindows()
