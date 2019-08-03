import cv2
import numpy as np

#왼쪽라인 보정상수
line_constant1 = -10
#오른쪽라인 보정상수
line_constant2 = +10
#조사창 최대 픽셀 수
max_pixel_num = 233
#조사창 최소 픽셀 수
min_pixel_num = 10
#픽셀 평행이동 상수
shift_pixel = 102
#다항식 차수
poly_order = 2
#차이 상수(아래 점 포인트와 위의 점 포인트 간 최대 픽셀 거리차)
del_cons = 30

#이전 조사창 중점 위치 저장하는 배열, 조사창 개수가 바뀔 때 변경해줘야함
x1_points1 = []
x1_points1 = [None for _ in range(3)]
x1_points2 = []
x1_points2 = [None for _ in range(3)]
x1_points3 = []
x1_points3 = [None for _ in range(3)]
x2_points1 = []
x2_points1 = [None for _ in range(3)]
x2_points2 = []
x2_points2 = [None for _ in range(3)]
x2_points3 = []
x2_points3 = [None for _ in range(3)]

def draw_points(img, x_points, y_points, color, thickness=3):
    if(len(x_points) != len(y_points)):
        print("error1")
        return
    try:
        for i in range(len(x_points)):
            if(x_points[i]!=None):
                cv2.line(img, (int(x_points[i]), int(y_points[i])), (int(x_points[i]), int(y_points[i])),
                         color, thickness=9)
    except:
        print("error2")

def make_route_points(img, n_windows, x1_default = 1300, x2_default = 270):
    #cv2.imshow('img', img)
    h, w = img.shape[:2]
    result = np.zeros_like(img)
    
    y_points = []
    # 조사창의 간격 정하기
    interval_y = int(h/(n_windows*2))
    interval_x = int(w/50)
    y=interval_y
    difference=0

    # 조사창들의 y값들 정하기
    for i in range(n_windows):
        y_points.append(y)
        y += interval_y * 2
    if (y_points[-1]+interval_y)>h :
        difference = y_points[-1]+interval_y-h
        y_points[-1]-=difference

    # 좌우 라인 분할
    img1 = img[0: h, 0: int(w/2)]
    img2 = img[0: h, int(w/2): w]
    left_end = img1.shape[1]
    right_end = img2.shape[1]

    # 왼쪽 라인 최대 x1 points 구하기
    x1_points=[]
    for n in range(0, n_windows):
        max_count = 0
        max_point = 0
        for i in range(left_end-interval_x-1, interval_x-1, -1):
            count = 0
            minimg = img1[y_points[n]-interval_y: y_points[n]+interval_y,
                          i-interval_x: i+interval_x]
            count = cv2.countNonZero(minimg)
            if(count>max_count):
                max_count = count
                max_point = i
            if(count>max_pixel_num):
                break
        if max_count == 0 or max_count < min_pixel_num:
            x1_points.append(None)
        else:
            x1_points.append((max_point+int(interval_x/2)+line_constant1))

    # 오른쪽 라인 최대 x2 points 구하기
    x2_points=[]
    for n in range(0, n_windows):
        max_count = 0
        max_point = 0
        for i in range(interval_x, right_end-interval_x):
            count = 0
            minimg = img2[y_points[n]-interval_y: y_points[n]+interval_y,
                          i-interval_x: i+interval_x]
            count = cv2.countNonZero(minimg)
            if(count>max_count):
                max_count = count
                max_point = i
            if(count>max_pixel_num):
                break
        if max_count == 0 or max_count < min_pixel_num:
            x2_points.append(None)
        else:
            x2_points.append(max_point+int(interval_x/2)+line_constant2)

    for i in range(n_windows):
        if(x2_points[i]!=None):
            x2_points[i]+=left_end

    #둘 중 하나가 None일 때 하나의 점을 평행이동 시켜 대입
    for i in range(n_windows):
        if(x1_points[i]==None and x2_points[i]!=None):
            x1_points[i]=x2_points[i]-shift_pixel
        elif(x1_points[i]!=None and x2_points[i]==None):
            x2_points[i]=x1_points[i]+shift_pixel
            
    #이전 조사창 위치를 저장하는 배열을 이용해 총 3개의 위치 평균
    global x1_points1, x1_points2, x1_points3, x2_points1, x2_points2, x2_points3        
    x1_points3 = x1_points2
    x1_points2 = x1_points1
    x1_points1 = x1_points
    for i in range(len(x1_points)):
        count = 0
        if(x1_points1[i] != None):
            count += 1
        else:
            x1_points1[i] = x1_default
            x1_points[i] = x1_points1[i]
            count += 1
        if(x1_points2[i] != None):
            x1_points[i]+=x1_points2[i]
            count += 1
        if(x1_points3[i] != None):
            x1_points[i]+=x1_points3[i]
            count += 1
        if(count!=0):
            x1_points[i] = int(x1_points[i]/count)
                
    x2_points3 = x2_points2
    x2_points2 = x2_points1
    x2_points1 = x2_points
    for i in range(len(x2_points)):
        count = 0
        if(x2_points1[i] == None):
            x2_points[i]= x2_default
            count +=1
        else:
            count += 1
        if(x2_points2[i] != None):
            x2_points1[i]+=x2_points2[i]
            count += 1
        if(x1_points3[i] != None):
            x2_points1[i]+=x2_points3[i]
            count += 1
        if(count!=0):
            x2_points[i] = int(x2_points1[i]/count)

    for i in range(len(x1_points)-1):
        if(abs(x1_points[len(x1_points)-i-1] - x1_points[len(x1_points)-i-2])
           > del_cons):
            x1_points[len(x1_points)-i-2] = x1_points[len(x1_points)-i-1]
            
    for i in range(len(x2_points)-1):
        if(abs(x2_points[len(x2_points)-i-1] - x2_points[len(x2_points)-i-2])
           > del_cons):
            x2_points[len(x2_points)-i-2] = x2_points[len(x2_points)-i-1]
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)    
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_points(img, x1_points, y_points, [0, 255, 0])
    draw_points(img, x2_points, y_points, [0, 0, 255])

    img = draw_lane(img, x1_points, x2_points, y_points)

    cv2.imshow('result', img)

def draw_lane(img, x1_points, x2_points, y_points):
    left_x=[]
    left_y=[]
    for i in range(len(x1_points)):
        if(x1_points[i] != None):
            left_x.append(x1_points[i])
            left_y.append(y_points[i])

    right_x=[]
    right_y=[]
    for i in range(len(x2_points)):
        if(x2_points[i] != None):
            right_x.append(x2_points[i])
            right_y.append(y_points[i])
            
    left_fit = np.polyfit(left_y, left_x, poly_order)
    right_fit = np.polyfit(right_y, right_x, poly_order)
    
    left = np.poly1d(left_fit)
    right = np.poly1d(right_fit)
    x_left = np.linspace(0,1440,1440)
    y_left = np.round(left(x_left))
    x_right = np.linspace(0,1440,1440)
    y_right = np.round(right(x_right))
    draw_points(img, y_left, x_left, [255,255,0], thickness=2)
    draw_points(img, y_right, x_right, [0,255,255], thickness=2)

    return img
def make_LIDAR_points(img, lidar):
    h, w = img.shape[:2]
      
    rx=[]
    ry=[]
    lidar2=[]

    #미터가 픽셀 단위로 변경된 라이더 값
    for i in range(361):
        lidar2.append(lidar[i]*100) #일단 1픽셀 당 1cm로 간주하기로 약속
    
    for i in range(361):
        angle = -((i/2)*np.pi/180 +np.pi/2)
        rx.append(lidar2[i] * np.sin(angle)+w/2)
        ry.append(lidar2[i] * np.cos(angle)+h)
        
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_points(img, rx, ry, [0, 0, 255])

    cv2.imshow("LIDAR", img)
