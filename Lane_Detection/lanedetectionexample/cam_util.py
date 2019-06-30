import cv2
import numpy as np
import os


def get_img(UDPSocket_cam, params_cam):
    '''
    receive a camera image
    \n UDPSocket_cam : UDP server socket
    \n params_cam : parameters from cameras 
    '''

    UnitBlockSize_cam = params_cam["Block_SIZE"]
    max_len = np.floor(params_cam["WIDTH"]*params_cam["HEIGHT"]/UnitBlockSize_cam/2)-1
    
    TotalBuffer = []
    num_block = 0

    while True:

        bytesAddressPair = UDPSocket_cam.recvfrom(UnitBlockSize_cam)
        UnitBlock = bytesAddressPair[0]
        
        UnitIdx = np.frombuffer(UnitBlock[3:7], dtype = "int")[0]
        UnitSize = np.frombuffer(UnitBlock[7:11], dtype = "int")[0]
        UnitTail = UnitBlock[-2:]
        UnitBody = np.frombuffer(UnitBlock[11:(11 + UnitSize)], dtype = "uint8")
        
        if num_block == UnitIdx:
            TotalBuffer.append(UnitBody)
            num_block += 1
        else:
            TotalBuffer = []
            num_block = 0

        if UnitTail==b'EI' and len(TotalBuffer)>max_len:

            TotalIMG = cv2.imdecode(np.hstack(TotalBuffer), 1)

            break

    return TotalIMG


def cam_calib(params_calib):
    '''
    calibrate a camera 
    \n params_cam : parameters from cameras 
    '''
    
    corner_rows = params_calib["CORNER ROWS"]
    corner_cols = params_calib["CORNER COLS"]

    #prepare image folder directory to load images
    PATH_TO_TEST_IMAGES_DIR = params_calib["DIR FOLDER"]
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, str(i).zfill(2)+'.jpg') for i in range(1, 1+params_calib["# of img"])]

    #termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001)

    #prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(CORNER ROWS-1,CORNER COLS-1,0)
    objp = np.zeros((corner_cols*corner_rows,3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_rows,0:corner_cols].T.reshape(-1,2)
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for image_path in TEST_IMAGE_PATHS:

        xyz_img = cv2.imread(image_path)
        h, w = xyz_img.shape[:2]
        xyz_gray = cv2.cvtColor(xyz_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(xyz_gray, (corner_rows, corner_cols), None)

        if ret == True:
            objpoints.append(objp)

            cv2.cornerSubPix(xyz_gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            #draw and display the corners

            if params_calib["VIEW CHECKBOARD CALIB"]:
                cv2.drawChessboardCorners(xyz_img, (corner_rows, corner_cols), corners, ret)
                cv2.imshow(image_path, cv2.resize(xyz_img, (w, h),
                                                    interpolation=cv2.INTER_LINEAR))
                cv2.waitKey(params_calib["VIEW TIME"])
                cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, xyz_gray.shape[::-1],None,None)

    return ret, mtx, dist, rvecs, tvecs


def cam_optimalmatrix(mtx, dist, params_cam):
    '''
    get the optimal camera matrix  
    \n mtx : instric camera matrix
    \n dist : distortion coefficient
    \n params_cam : parameters from cameras 
    '''
    w,  h = params_cam["WIDTH"], params_cam["HEIGHT"]
    newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    return newcameramtx, roi


def undistort_cam_img(img, mtx, dist, newcameramtx, roi, crop=False):
    '''
    get the optimal camera matrix  
    \n img : camera image array
    \n mtx : instric camera matrix
    \n newcameramtx : optimal calibrated camera matrix
    \n roi : range of valid image
    \n crop : crop the invalid part of the undistorted image 
    '''
    
    # undistort
    undist = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    if crop:
        x,y,w,h = roi
        undist = undist[y:y+h, x:x+w]
    
    return undist