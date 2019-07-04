import sys
import os

sys.path.remove("/home/junho/catkin_ws/devel/lib/python2.7/dist-packages")
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import random
import numpy as np

index = 1
lightness = 4
size = 5

'''
target_path = "./target/%d-1.png"%(j+1)
background_path = "./background"
'''

for j in range(1):
    target2 = cv2.imread("./target/%d.png"%(j+1))
    for i in range(1,size+1):
        for (path, dir, files) in os.walk("./background"):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.jpg':
                    real_path = path+'/'+filename
                    background = cv2.imread(real_path)

                    target = cv2.resize(target2,None,fx=0.35*i,fy=0.35*i,interpolation = cv2.INTER_LINEAR)
                    background = cv2.resize(background,dsize=(920,920),interpolation = cv2.INTER_LINEAR)
                    rows, cols, channels = target.shape
                    rand_row = int((len(background[:,0,0])-rows)*random.random())
                    rand_col = int((len(background[0,:,0])-cols)*random.random())

                    roi = background[rand_row:rows+rand_row,rand_col:cols+rand_col]

                    img2gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)
                    #if target's background is white use cv2.THRESH_BINARY_INV
                    mask_inv = cv2.bitwise_not(mask)

                    img1_fg = cv2.bitwise_and(target, target, mask=mask)
                    img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    dst = cv2.add(img1_fg,img2_bg)
                    background[rand_row:rows+rand_row,rand_col:cols+rand_col] = dst

                    y_center = (rand_row+rows/2)/len(background[:,0,0])
                    x_center = (rand_col+cols/2)/len(background[0,:,0])
                    y_width = rows/len(background[:,0,0])
                    x_width = cols/len(background[0,:,0])

                    filename = os.path.splitext(filename)[0]
                    filename = str(int(filename)+200*(i-1))+'.jpg'

                    cv2.imwrite("/home/junho/Autonomous/HENVEN_PathPlanning/Yolo/merge/%s"%filename,background)
                    f = open("/home/junho/Autonomous/HENVEN_PathPlanning/Yolo/merge/%s.txt"%os.path.splitext(filename)[0], 'w')
                    f.write("%d %.4f %.4f %.4f %.4f" % (j, x_center, y_center, x_width, y_width))
                    f.close()


'''
f = open("./train.txt", 'w')   
for i in range(1,3001):
    f.write("data/img/%d.jpg\n"%i)
f.close()

'''