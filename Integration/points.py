import cv2
import numpy as np

def draw_points(img, x_points, y_points, color, thickness=3):
    if(len(x_points) != len(y_points)):
        print("error1")
        return
    try:
        for i in range(len(x_points)):
            cv2.line(img, (int(x_points[i]), int(y_points[i])), (int(x_points[i]), int(y_points[i])),
                     color, thickness=3)
    except:
        print("error2")

def make_route_points(img, n_windows, x1_default = 110, x2_default =120):
    #cv2.imshow('img', img)
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
    draw_points(img1, x1_points, y_points, [0, 255, 0])
    draw_points(img2, x2_points, y_points, [0,255,0])
    draw_points(img, x_points, y_points, [0,255,0])
    
    cv2.imshow('result', img)
    #cv2.imshow('result_left', img1)
    #cv2.imshow('result_right', img2)
    
def make_LIDAR_points(img, lidar):
    h, w = img.shape[:2]
      
    rx=[]
    ry=[]
    lidar2=[]

    #미터가 픽셀 단위로 변경된 라이더 값
    for i in range(361):
        #lidar2.append(lidar[i]*280/3.5) #280픽셀당 3.5m
        lidar2.append(lidar[i]*100) #일단 1픽셀 당 1cm로 간주하기로 약속
    
    for i in range(361):
        angle = -((i/2)*np.pi/180 +np.pi/2)
        rx.append(lidar2[i] * np.sin(angle)+w/2)
        ry.append(lidar2[i] * np.cos(angle)+h)
        
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_points(img, rx, ry, [0, 0, 255])

    cv2.imshow("LIDAR", img)

