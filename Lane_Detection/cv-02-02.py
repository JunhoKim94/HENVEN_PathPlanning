import numpy as np
import cv2

def showVideo():
    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture(1)
        cap2 = cv2.VideoCapture(2)
    except:
        print('카메라 구동 실패')
        return

    fps=20.0
    width = int(cap.get(3))
    height = int(cap.get(4))
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    out = cv2.VideoWriter('cam_unit3.avi', fcc, fps, (width, height))
    out2 = cv2.VideoWriter('cam_unit4.avi', fcc, fps, (width, height))
    print('녹화를 시작합니다.')
    
    while True:
        ret, frame = cap.read()
        ret, frame2 = cap2.read()

        if not ret:
            print('비디오 읽기 오류')
            break

        cv2.imshow('video', frame)
        cv2.imshow('video2', frame2)
        out.write(frame)
        out2.write(frame2)

        k = cv2.waitKey(1)
        if k == 27:
            print('녹화를 종료합니다')
            break

    cap.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

showVideo()
