import numpy as np
import cv2
import matplotlib.pyplot as plt
import channelplus as cp
import Image_util as iu

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
    #480, 360
    pts1 = np.float32([(275, 240), (403, 240), (527, 332), (151, 332)])
    pts2 = np.float32([(550, 0), (890, 0), (860, 480), (580, 480)])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    tran_img = cv2.warpPerspective(img, M, (1440,480))
    #720, 480

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

def make_points(img, n_windows, x1_default = 110, x2_default =120):
    cv2.imshow('img', img)
    h, w = img.shape[:2]
    result = np.zeros_like(img)
    
    y_points = []
    # 조사창의 간격 정하기
    interval_y = int(h/(n_windows*2))
    interval_x = int(w/10)
    y=interval_y
    difference=0

    # 조사창들의 y값들 정하기
    for i in range(n_windows):
        y_points.append(y)
        y += interval_y * 2
    if (y_points[-1]+interval_y)>h :
        difference = y_points[-1]+interval_y-h
        y_points[-1]-=difference

    # 좌우 라인 분할
    img1 = img[0: h, 0: int(w/2)]
    img2 = img[0: h, int(w/2): w]
    left_end = img1.shape[1]
    right_end = img2.shape[1]

    # 왼쪽 라인 최대 x1 points 구하기
    x1_points=[]
    for n in range(0, n_windows):
        max_count = 0
        max_point = 0
        for i in range(interval_x, left_end-interval_x):
            count = 0
            minimg = img1[y_points[n]-interval_y: y_points[n]+interval_y,
                          i-interval_x: i+interval_x]
            count = cv2.countNonZero(minimg)
            if(count>max_count):
                max_count = count
                max_point = i
        if max_count == 0:
            x1_points.append(x1_default)
        else:
            x1_points.append((max_point+int(interval_x/2)+15))

    # 오른쪽 라인 최대 x2 points 구하기
    x2_points=[]
    for n in range(0, n_windows):
        max_count = 0
        max_point = 0
        for i in range(interval_x, right_end-interval_x):
            count = 0
            minimg = img2[y_points[n]-interval_y: y_points[n]+interval_y,
                          i-interval_x: i+interval_x]
            count = cv2.countNonZero(minimg)
            if(count>max_count):
                max_count = count
                max_point = i
        if max_count == 0:
            x2_points.append(x2_default)
        else:
            x2_points.append(max_point+int(interval_x/2)+15)

    # 두 포인트 합쳐서 중간 값 만들기
    x_points=[]
    for i in range(n_windows):
        x_points.append((x1_points[i]+x2_points[i]+left_end)/2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)    
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_points(img1, x1_points, y_points)
    draw_points(img2, x2_points, y_points)
    draw_points(img, x_points, y_points)
    cv2.imshow('result', img)
    cv2.imshow('result_left', img1)
    cv2.imshow('result_right', img2)
    
#img = cv2.imread('C:\photo\images/blue_lane.jpg')
video="C:\photo\images/challenge.avi"
cap = cv2.VideoCapture(video)
while True:
    ret, img = cap.read()
    if not ret:
        print('비디오 끝')
        break
    cv2.waitKey(5)
    img = cp.channelplus(img)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = smoothing(img)
    cv2.imshow('wow', img)
    binary_img = bin_img(img)
    cv2.imshow('bin', binary_img)
    masked_img = reg_of_int(binary_img)
    tran_img=perspective_transform(masked_img)
    cv2.imshow('result', tran_img)
    make_points(tran_img, 10)

cv2.waitKey(0)
cv2.destroyAllWindows()

