

import numpy as np
import cv2


import socket
import time

from cam_util import get_img
from lane_util import warp_image, binary_pipeline, morphologic_process, get_poly_val, fit_track_lanes, \
    callback_ths, crop_points, draw_lane_img, visualize_images

params_cam = {
    "WIDTH": 480, # image width
    "HEIGHT": 320, # image height
    "FOV": 90, # Field of view
    "localIP": "127.0.0.1",
    "localPort": 1232,
    "Block_SIZE": int(65000),
    "X": 0, # meter
    "Y": 0,
    "Z": 1.5,
    "YAW": 0, # deg
    "PITCH": 0,
    "ROLL": 0
}

UDP_cam = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDP_cam.bind((params_cam["localIP"], params_cam["localPort"]))


poly_order = 1

x = params_cam["WIDTH"]
y = params_cam["HEIGHT"]


source_points = np.float32([
    [0.05 * x, 0.75 * y],
    [(0.5 * x) - (x*0.13), (0.52)*y],
    [(0.5 * x) + (x*0.13), (0.52)*y],
    [x - (0.05 * x), 0.75 * y]
    ])
'''
source_points = np.float32([
    [0.01 * x, 0.9 * y],
    [(0.5 * x) - (x*0.09), (0.52)*y],
    [(0.5 * x) + (x*0.09), (0.52)*y],
    [x - (0.01 * x), 0.9 * y]
    ])
'''


def main():

    # initial threshold
    th_light = 75
    
    cv2.namedWindow('Result')
    cv2.createTrackbar('Luminance:', 'Result', th_light, 99, callback_ths)
    
    
    for _ in range(20000):

        # record time
        t_s = time.time()
        
        # image parsing
        img_cam = get_img(UDP_cam, params_cam)
        
        img_binary = binary_pipeline(img_cam, th_light, (80, 95)) #lane
        img_binary_warp = warp_image(img_binary, source_points)
        img_morph = morphologic_process(img_binary_warp, 5, 1, 1)
        
        t_cost = time.time()-t_s

        fit_check, left_fit, right_fit = fit_track_lanes(img_morph, poly_order=poly_order,
                                                                    nwindows=30,
                                                                    margin=30,
                                                                    minpix=20)

        img_morph = cv2.cvtColor(img_morph*255, cv2.COLOR_GRAY2BGR)

        if fit_check:

            ploty = np.linspace(0, params_cam["HEIGHT"]-1, params_cam["HEIGHT"])
            left_fitx = get_poly_val(ploty, left_fit)
            right_fitx = get_poly_val(ploty, right_fit)

            left_fitx, left_fity = crop_points(left_fitx.astype(np.int32), ploty.astype(np.int32), params_cam["WIDTH"], params_cam["HEIGHT"])
            right_fitx, right_fity = crop_points(right_fitx.astype(np.int32), ploty.astype(np.int32), params_cam["WIDTH"], params_cam["HEIGHT"])
        
            img_line = draw_lane_img(img_morph, left_fitx, left_fity, right_fitx, right_fity)

        else:
            img_line = np.copy(img_morph)

        th_light = visualize_images([cv2.polylines(img_cam,[source_points.astype(np.int32)],True,(255,0,0), 5),
                                    cv2.cvtColor(img_binary*255, cv2.COLOR_GRAY2BGR),
                                    img_morph,
                                    img_line],
                                    t_cost, params_cam, img_name=['src', 'binary', 'morph', 'lane'])

    UDP_cam.close()


if __name__ == '__main__':

    main()