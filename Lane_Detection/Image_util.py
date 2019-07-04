'''
import sys
sys.path.remove("/home/junho/catkin_ws/devel/lib/python2.7/dist-packages")
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
'''
import cv2
import numpy as np

def Gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    img = np.uint8(img*255)
    return img

def Warp_Image(img, source_points):
    
    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]
    
    destination_points = np.float32([
    [0, y],
    [0, 0],
    [x, 0],
    [x, y]
    ])
    
    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    
    warped_img = cv2.warpPerspective(img, perspective_transform, image_size, flags=cv2.INTER_LINEAR)
    
    return warped_img

def rotate_Image(image,deg):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

#for eliminate Noise
def Opening_Image(image,kernel_size,n_e,n_d,flag=cv2.MORPH_RECT):
    kernel = cv2.getStructuringElement(flag,(kernel_size, kernel_size))
    #cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS, cv2.MORPH_RECT
    img = cv2.erode(image, kernel, iterations = n_e)
    img = cv2.dilate(image, kernel, iterations = n_d)
    return img

#for smoothing
def Closing_Image(image,kernel_size,n_e,n_d):
    kernel = cv2.getStructuringElemnet(cv2.MORPH_RECT,(kernel_size,kernel_size))
    img = cv2.dilate(image,kernel,iterations=n_d)
    img = cv2.erode(image,kernel,iteration= n_e)
    return img