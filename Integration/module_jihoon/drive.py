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

video="./test.mp4"
cap = cv2.VideoCapture(video)

while True:
    ret, img = cap.read()
    if not ret:
        print('비디오 끝')
        break
    img = cv2.resize(img,dsize = (854,480),interpolation = cv2.INTER_AREA)
    img1 = iu.Make_Binary(img)
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #cv2.waitKey(1)
    cv2.imshow("fff",img1)
    img2 =cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    pts.make_LIDAR_points(img1, lidar)
    pts.make_route_points(img1, 3)
    if cv2.waitKey(1) & 0xFF == 27:
        break        

cap.release()
cv2.destroyAllWindows()