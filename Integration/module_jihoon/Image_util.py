import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
boundaries = [
    (np.array([161, 155, 84], dtype="uint8"), np.array([179, 255, 255], dtype="uint8")), # red1
    (np.array([0, 100, 70], dtype="uint8"), np.array([20, 255, 255], dtype="uint8")), # red2
    (np.array([94, 80, 200], dtype="uint8"), np.array([126, 255, 255], dtype="uint8")), # blue
    (np.array([0, 60, 100], dtype="uint8"), np.array([51, 255, 255], dtype="uint8")), #yellow
    (np.array([200, 0, 140], dtype="uint8"), np.array([255, 255, 255], dtype="uint8")) # white
]

line_colors = [
    [0, 0, 255], # red
    [255, 0, 0], # blue
    [0, 255, 255], # yellow
    [255, 255, 255] # white
]

kernel = np.ones((5,5), np.uint8)

# points for perspective_transform
pts1 = np.float32([(536, 471), (247, 651), (1094, 651), (771, 471)])
pts2 = np.float32([(480, 0), (480, 1000), (960, 1000), (960, 0)])

# 디스플레이 창 크기
display = (1440, 960)

# roi 범위 설정
vertics = np.array([[(536, 482), (247, 651), (1094, 651), (771, 471)]]
                       ,np.int32)
#720, 960
def Warp_Image(img):
    image_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped_img = cv2.warpPerspective(img, M, display)
    return warped_img

def reg_of_int(img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertics, (255,255,255))
    masked = cv2.bitwise_and(img, mask)
    return masked

def DetectColor(img, color): # color = b, r, y, w
    minRange, maxRange = 0, 0
    if color == "w":
        (minRange, maxRange) = boundaries[4]
        mask = cv2.inRange(img, minRange, maxRange)
    elif color == "y":
        (minRange, maxRange) = boundaries[3]
        mask = cv2.inRange(img, minRange, maxRange)
    elif color == "b":
        (minRange, maxRange) = boundaries[2]
        mask = cv2.inRange(img, minRange, maxRange)
    elif color == "r":
        (minRange, maxRange) = boundaries[0]
        mask = cv2.inRange(img, minRange, maxRange)
        (minRange, maxRange) = boundaries[1]
        mask = mask + cv2.inRange(img, minRange, maxRange)
    else:
        print("In Image_util.py DetectColor - Wrong color Argument")
    return mask


def Gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    img = np.uint8(img*255)
    return img


def CloseImage(img):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def OpenImage(img):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def Make_Binary(img):
    img = reg_of_int(img)
    img = Gamma_correction(img, 2)
    warp_img = Warp_Image(img)
    img1 = np.zeros_like(warp_img)
    img2 = DetectColor(warp_img, 'b')
    img2=cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img1 = cv2.add(img1, img2)
    img2 = DetectColor(warp_img, 'y')
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img1 = cv2.add(img1, img2)
    img2 = DetectColor(warp_img, 'w')
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img1 = cv2.add(img1, img2)
    
    img_ret = OpenImage(img1)
    return img_ret
