import numpy as np
import cv2
import channelplus as cp
import points as pts
import perspective as pp 

lidar = np.zeros(361) #0.5도씩 361개 -> 180도 표현
lidar[0:120] = 8
lidar[120:180] = 5
lidar[260:360] = 7


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

  
video="C:\\Users\\ybin0\\Desktop\\lane/challenge.mp4"
cap = cv2.VideoCapture(video)
while True:
    ret, img = cap.read()
    
    if not ret:
        print('비디오 끝')
        break
    
    cv2.waitKey(10)
    img = cp.channelplus(img)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #cv2.imwrite("05img.jpg",img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = smoothing(img)
    binary_img = bin_img(img)
    #cv2.imshow('bin', binary_img)
    masked_img = reg_of_int(binary_img)
    tran_img=pp.perspective_transform(masked_img)
    #cv2.imshow('result', tran_img)
    pts.make_LIDAR_points(tran_img, lidar)
    pts.make_route_points(tran_img, 10)
    

cv2.waitKey(0)
cv2.destroyAllWindows()
