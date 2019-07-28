import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import Image_util as iu
import points as pts

lidar = np.zeros(361) #0.5도씩 361개 -> 180도 표현
lidar[0:120] = 8
lidar[120:180] = 5
lidar[260:360] = 7

video="C:\photo\images/cam_2_short.mp4"
cap = cv2.VideoCapture(video)

while True:
    ret, img = cap.read()
    if not ret:
        print('비디오 끝')
        break
    cv2.waitKey(10)
    img1 = iu.Make_Binary(img)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 =cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    pts.make_LIDAR_points(img1, lidar)
    pts.make_route_points(img1, 15)

cv2.waitKey(0)
cv2.destroyAllWindows()
