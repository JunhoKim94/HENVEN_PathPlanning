import numpy as np
import cv2


def callback_ths(x):
    return x


def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    img = np.uint8(img*255)
    return img


def binary_pipeline(img, th_L, hrange):
    '''
    binarize an image for lane detection
    \n img : source image
    \n th_L : luminance percentile threshold
    \n hrange : threshold hue range of yellow lanes
    '''

    # convert to HLS color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # split images to the H, L channel
    H, L, _ = cv2.split(hls_img)

    # define a luminance threshold to the H channel
    lthresh=(np.percentile(L.reshape([-1]), th_L), 255)
    
    # return a binary image of threshold result
    binary_output1 = np.zeros_like(L)
    binary_output1[(H > hrange[0]) & (H <= hrange[1])] = 1

    binary_output2 = np.zeros_like(L)
    binary_output2[(L > lthresh[0]) & (L <= lthresh[1])] = 1

    # combine two binary results
    binary_output = cv2.bitwise_or(binary_output1, binary_output2)
    return binary_output


def morphologic_process(img, kernel_size, n_e, n_d):

    # kernel for morphing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

    img = cv2.erode(img, kernel, iterations = n_e)
    img = cv2.dilate(img, kernel, iterations = n_d)

    return img


def get_poly_val(y,poly_coeff):
    val = 0
    len_poly = len(poly_coeff)

    for i in range(len_poly):
        val += poly_coeff[i]*y**(len_poly-1-i)
    
    return val


def warp_image(img, source_points):
    
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


def empty_check(leftx, lefty, rightx, righty):

    return leftx.any() and lefty.any() and rightx.any() and righty.any()


def fit_track_lanes(binary_warped, poly_order, nwindows, margin, minpix):
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # we need max for each half of the histogram. the example above shows how
    # things could be complicated if didn't split the image in half 
    # before taking the top 2 maxes
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    # this will throw an error in the height if it doesn't evenly divide the img height
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    # Set minimum number of pixels found to recenter window
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    
    if empty_check(leftx, lefty, rightx, righty):

        # Fit a polynomial to each
        left_fit = np.polyfit(lefty, leftx, poly_order)
        right_fit = np.polyfit(righty, rightx, poly_order)

        return empty_check(leftx, lefty, rightx, righty), left_fit, right_fit

    else:

        return empty_check(leftx, lefty, rightx, righty), [], []


def visualize_images(img_in, t_cost, params_cam, img_name):
    '''
    place the lidar points into numpy arrays in order to make intensity map
    \n img_in : image list for comparison
    \n t_cost : inference time
    \n params_cam : camera parameters
    \n img_name : image name list
    '''

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    num_img = len(img_name)

    silver1= np.zeros((80, params_cam["WIDTH"]*num_img, 3), np.uint8)
    silver1[:] = (211,211,211)

    for i in range(num_img):
        cv2.putText(silver1, img_name[i] ,(30+i*params_cam["WIDTH"],50), font, 2,(0,0,0), 3, 0)
    
    silver2= np.zeros((60, params_cam["WIDTH"]*num_img, 3), np.uint8)
    silver2[:] = (211,211,211)
    cv2.putText(silver2,'Inference time: {:.4f}s'.format(t_cost),(30,30), font, 1,(0,0,0), 2, 0)
    
    img_hcat = cv2.hconcat((img_in[0], img_in[1]))

    for i in range(2, num_img):
        img_hcat = cv2.hconcat((img_hcat, img_in[i]))

    img_vcat1 = cv2.vconcat((silver1, img_hcat))
    img_vcat2 = cv2.vconcat((img_vcat1, silver2))
    
    cv2.imshow('Result', img_vcat2)
    cv2.waitKey(1)
    
    th_L = cv2.getTrackbarPos('Luminance:','Result')

    return th_L


def draw_lane_img(img, leftx, lefty, rightx, righty):
    '''
    place the lidar points into numpy arrays in order to make intensity map
    \n img : source image
    \n leftx, lefty, rightx, righty : curve fitting result 
    '''
    point_np = np.copy(img)

    #Left Lane
    point_np[lefty, leftx,0] = 0
    point_np[lefty, leftx,1] = 0
    point_np[lefty, leftx,2] = 255

    #Right Lane
    point_np[righty, rightx,0] = 255
    point_np[righty, rightx,1] = 0
    point_np[righty, rightx,2] = 0

    return point_np


def crop_points(xi, yi, img_w, img_h):
    '''
    crop the lidar points on images within width and height
    \n xi, yi : xy components of lidar points w.r.t a 2d plane
    \n img_w, img_h : a width and a height of a image from a camera
    '''
    #cut the lidar points out of width

    xi_crop, yi_crop = np.copy(xi), np.copy(yi)

    crop_x_max_idx = np.where(xi_crop<img_w)[0] 

    xi_crop = xi_crop[crop_x_max_idx]
    yi_crop = yi_crop[crop_x_max_idx]

    crop_x_min_idx = np.where(xi_crop>=0)[0]

    xi_crop = xi_crop[crop_x_min_idx]
    yi_crop = yi_crop[crop_x_min_idx]
    
    #cut the lidar points out of height
    crop_y_max_idx = np.where(yi_crop<img_h)[0]

    xi_crop = xi_crop[crop_y_max_idx]
    yi_crop = yi_crop[crop_y_max_idx]

    crop_y_min_idx = np.where(yi_crop>=0)[0]

    xi_crop = xi_crop[crop_y_min_idx]
    yi_crop = yi_crop[crop_y_min_idx]
    
    return xi_crop, yi_crop