#Don't using this sys for your computers
'''
import sys
sys.path.remove("/home/junho/catkin_ws/devel/lib/python2.7/dist-packages")
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
'''

#start here

import numpy as np
import cv2
import Image_util as iu

path = "./Lane_Detection/Lane_image/blue_lane.jpg"

img_o = cv2.imread(path)
img = cv2.resize(img_o,dsize=(480,380),interpolation=cv2.INTER_LINEAR)
img_brightness = iu.Gamma_correction(img, 5)
#Gaussian Blur for noise
#img = cv2.GaussianBlur(img, (5,5),2)

b,g,r = cv2.split(img)
#yellow = np.array((r+g),dtype=np.uint8)


img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, ori = cv2.threshold(img,150,255,cv2.THRESH_BINARY)

#Threshold
ret , b_th = cv2.threshold(b,180,255,cv2.THRESH_BINARY)
ret , g_th = cv2.threshold(g,180,255,cv2.THRESH_BINARY)
ret , r_th = cv2.threshold(r,220,255,cv2.THRESH_BINARY)

#adaptiveThreshold
r_ad = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
g_ad = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
b_ad = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)

#concat image for comparing
img_concat = cv2.hconcat((r_th,g_th,b_th))
img_concat_ad = cv2.hconcat((r_ad,g_ad,b_ad))

#bitwise calculation
img_bit = cv2.bitwise_or(b_th,g_th)
img_bit = cv2.bitwise_or(img_bit,r_th)

img_concat_rgb = cv2.hconcat((ori,img_bit,img))

cv2.imshow("Window",img_brightness)
cv2.waitKey(0)
cv2.destroyAllWindows()
