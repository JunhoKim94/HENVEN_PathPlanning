"""
Created on Sun Jul 21 12:56:00 2019

@author: kypark
"""
import os
import cv2
import lane
import lidar
import socket
import numpy as np


def make_points(img, n_windows, x1_default = 110, x2_default =120):
    #cv2.imshow('img', img)
    h, w = img.shape[:2]
    result = np.zeros_like(img)
    print(h, w)
    
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
    lane.draw_points(img1, x1_points, y_points)
    lane.draw_points(img2, x2_points, y_points)
    lane.draw_points(img, x_points, y_points)
    cv2.imshow('result', img)
    #cv2.imshow('result_left', img1)
    #cv2.imshow('result_right', img2)



'''
< main function > : 카메라 + lidar의 값을 출력함.

카메라 차선인식에 임의로 라이다 값을 패딩해서 씌워놓음.
지금은 라이다 값이 파일로 있는데 스트리밍으로 받을 경우 수정해함야 함.
스트리밍으로 받을 경우 camera와 lidar 각각에서 받는 영상의 시간차를 조절해야 함 -> 직접 촬영한 결과물로 시간 조절해야 할듯.
camera와 lidar 각각에서 받는 물체의 상대적 위치를 계산해서 lidar 값을 조정해야 함 -> 직접 촬영한 결과물로 조절해야 할듯.
lane.py, lidar.py 파일에 각각에 사용되는 함수들을 모듈화 시켜놓음.

'''

########## open files ##########
## camera (for lane detection)
video_path = os.path.join('..', '..', 'Lane_Detection', 'Lane_image', 'test.mp4') ## for window & linux
video_cap = cv2.VideoCapture(video_path)

## lidar (for object detection) : socket 통신으로 가져옴옴
HOST = '127.0.0.1'
PORT = 10018
BUFF = 57600
RAD = 500

sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_lidar.connect((HOST, PORT))
################################



while True:
    ret, img = video_cap.read()
    
    if not ret:
        print('비디오 끝')
        break
    
    ######## camera 부분 영상처리 #######
    cv2.waitKey(10)
    img = img2 =cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) ## 이미지 resizing
    # cv2.imshow("img", img)
    # cv2.imwrite("05img.jpg",img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## 이미지를 grayscale로 바꿈
    img = lane.smoothing(img) ## smoothing 처리

    ## 이미지를 bird view로 변환
    binary_img = lane.bin_img(img)
    # cv2.imshow('bin', binary_img)
    masked_img = lane.reg_of_int(binary_img)
    tran_img = lane.perspective_transform(masked_img)

    ## lane 출력
    cv2.imshow('camera', tran_img)
    ####################################


    ######## lidar 부분 영상처리 ######## : camera쪽 영상처리 된 부분에 lidar를 덧입힘
    raw = sock_lidar.recv(BUFF).decode()
    ## 이부분은 나중에 영상 크기 정해지면 전역변수로 뺌!!

    if not raw.__contains__('sEA'):
        if cv2.waitKey(1) & 0xff == ord(' '): continue
        parsed_data = lidar.parsing_lidar(raw) ## 데이터를 파싱
        try:
            points = lidar.making_point_lidar(parsed_data) ## parsing되어 찍을 점을 저장
        except Exception: # lidar가 끝날경우 카메라만 계속 출력
            lidar.draw_lidar([(-100, -100)])
            make_points(tran_img, 10)
            lidar.draw_lidar_on_lane(tran_img, [(-100, -100)])
            continue
        
        lidar.draw_lidar(points) ## 빈 화면에 lidar 출력
    ####################################


    ########### lane + lidar ###########
    ## train된 이미지에 점을 찍음
    lidar.draw_lidar_on_lane(tran_img, points) ## lane 위에 points 출력

    ## 갈 수 있는 점 출력
    make_points(tran_img, 10)
    ####################################


cv2.waitKey(0)
cv2.destroyAllWindows()