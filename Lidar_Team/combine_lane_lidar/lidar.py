"""
Created on Sun Jul 21 16:23:00 2019

@author: kypark
"""

import numpy as np
import cv2
import threading
import time

# RAD = 320
PADDING = 5 ## 나중에 차 크기 측정해서 패딩 결정!!
TRANSFORM = 15 ## 나중에 차 크기 측정해서 camera <-> lidar 변환값 결정
width = 480
height = 320

## raw lidar file을 parsing해줌
def parsing_lidar(raw):
    return raw.split(' ')[116:477]

## parsing된 file에서 lidar 점을 추출해줌
def making_point_lidar(data): ## data: 찍을 점(parsing된 lidar 파일)
    current_frame = np.zeros((height, width), np.uint8)
    points = np.full((361, 2), -1000, np.int)  # 점 찍을 좌표들을 담을 어레이 (x, y), 멀리 -1000 으로 채워둠.

    data_list = [int(item, 16) for item in data]
    for theta in range(0, 361):
        r = data_list[theta] / TRANSFORM  # 차에서 장애물까지의 거리, 단위는 cm

        if 2 <= r:  # 라이다 바로 앞 1cm 의 노이즈는 무시

            # r-theta 를 x-y 로 바꿔서 (실제에서의 위치, 단위는 cm)
            x = r * np.cos(np.radians(0.5 * theta))
            y = r * np.sin(np.radians(0.5 * theta))
            # 좌표 변환, 화면에서 보이는 좌표(왼쪽 위가 (0, 0))에 맞춰서 집어넣는다
            points[theta][0] = round(x) + width / 2
            points[theta][1] = height - round(y)

    return points


## 추출된 점을 빈 화면에 출력함(lidar만)
def draw_lidar(points):
    current_frame = np.zeros((height, width), np.uint8)
    for point in points:  # 장애물들에 대하여
        cv2.circle(current_frame, tuple(point), PADDING, 255, -1)  # 캔버스에 점 찍기
    cv2.imshow("lidar", current_frame) ## lidar만 출력


## 추출된 점을 lane 위에 표시(lane + lidar)
def draw_lidar_on_lane(img, points): ## img: 점을 찍을 순간적인 lane frame
    for point in points:  # 장애물들에 대하여
        cv2.circle(img, tuple(point), PADDING, 255, -1)  # 캔버스에 점 찍기
    cv2.imshow("lane + lidar", img) ## lane + lidar
    return img