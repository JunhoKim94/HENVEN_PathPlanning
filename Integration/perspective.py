import numpy as np
import cv2

def perspective_transform(img):
    h, w = img.shape[:2]
    pts1 = np.float32([(275, 240), (403, 240), (527, 332), (151, 332)])
    pts2 = np.float32([(550, 20), (890, 20), (860, 500), (580, 500)])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    tran_img = cv2.warpPerspective(img, M, (1440,500))

    return tran_img
