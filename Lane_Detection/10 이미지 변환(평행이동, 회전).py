'''
    img2 = cv2.resize(img, None, fx=0.5, fy=1, interplation=cv2.INTER_AREA)

    None: dsize를 나타내는 튜플 값 (가로방향 픽셀 수, 세로방향 픽셀 수)로 나타냄
    fx, fy: 각각 가로 방향, 세로 방향으로 배율 인자 0.5로 지정하면 원래 크기의 0.5로 리사이징 하라는 의미
    interplolation: 리사이징을 수행할 때, 적용할 interpolation 방법

    INTER_NEAREST: 리사이징을 수행할 때 적용할 interpolation 방법 
    INTER_LINEAR: bilinear interpolation (디폴트 값)
    INTER_AREA: 픽셀 영역 관계를 이용한 resampling 방법으로 이미지 축소에 있어 선호되는 방법.
    이미지를 확대하는 경우에는 INTER_NEAREST와 비슷한 효과를 보임
    INTER_CUBIC: 4*4 픽셀에 적용되는 bicubic interpolation
    INTER_LANCZOS4: 8*8 픽셀에 적용되는 Lanzos interpolation

    INTER_CUBIC+INTER_LINEAR로 확대를 수행
'''

import numpy as np
import cv2

def transform():
    img = cv2.imread('C:\photo\images/image_curve.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    cv2.imshow('original', img)

    img2 =cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imshow('fx=0.5', img2)

    #x축으로 100이동 y축으로 50이동
    M = np.float32([[1,0,100], [0,1,50]])
    
    img3 = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('shift image', img3)

    #사진을 중심으로 각각 45도 90도 회전
    M1 = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
    M2 = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)  

    cv2.imshow('45-Rotated', img2)
    cv2.imshow('90-Rotated', img3)

transform()
