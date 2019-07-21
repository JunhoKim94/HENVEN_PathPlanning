import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def reg_of_int(img):
    img_h = img.shape[0]
    img_w = img.shape[1]

    # 범위 설정
    vertics = np.array([[(275, 240), (403, 240), (527, 332), (151, 332)]]
                       ,np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertics, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def bin_img(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    ret, bin_img = cv2.threshold(blur, 0, 255,
                                 cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bin_img

#CLAHE 알고리즘 적용
def smoothing(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(50,50))
    img2 = clahe.apply(img)
    return img2

def perspective_transform(img):
    h, w = img.shape[:2]
    pts1 = np.float32([(275, 240), (403, 240), (527, 332), (151, 332)])
    pts2 = np.float32([(70, 0), (410, 0), (380, 320), (100, 320)])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    tran_img = cv2.warpPerspective(img, M, (480,320))

    return tran_img

#low 와 high 비율은 1:2나 1:3 추천
def canny(img, low_thr, high_thr):
    img_cny=cv2.Canny(img, low_thr, high_thr)
    return img_cny

def draw_lines(img, lines, color=[0,0,255], thickness=2): #선 그리기
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1,y1), (x2,y2), color, thickness)
    except:
        print("error")

def draw_points(img, x_points, y_points, color=[0,255,0], thickness=3):
    if(len(x_points) != len(y_points)):
        print("error1")
        return
    try:
        for i in range(len(x_points)):
            cv2.line(img, (int(x_points[i]), int(y_points[i])), (int(x_points[i]), int(y_points[i])),
                     color=[0,255,0], thickness=3)
    except:
        print("error2")
        
#rho : hough space에서 원점으로 부터의 거리를 뜻하는 것(보통 1)
    # 얼마씩 증가시키면서 살필 것이지를 정함
#theta : 단위는 라디언 (도 * pi/ 180) 마찬가지로 보통 1이고
    # 얼마씩 증가시키면서 살필 것이지를 정함
#threshold : 직선으로 인식하는 최소 갯수(교점이 x-y에서 직선이됨)
#min_line_length : 선분의 최소 길이
#max_line_gap : 선 위의 점들 사이 최대 거리
#output은 선분의 시작점과 끝점이 됨
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    img_h, img_w = img.shape[:2]
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

def weighted_img(img, initial_img, α=1, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)