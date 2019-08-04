import cv2
import numpy as np
# 
'''
#################### PATH PLAN TEAM ####################
## ABOUT
- 차선을 인식해서 왼쪽차선, 오른쪽차선, 정지선의 위치를 반환해주는 코드
## INPUT & OUTPUT
- input: 카메라 원
- output: 각 미션 별 packet
'''
#다항식 차수
poly_order = 2
n_windows = 10
#조사창 너비 정하기 (이미지 좌우 크기/50)
windows_width = 50
#최대 픽셀 수
max_pixel_num = 600
#최소 픽셀 수
min_pixel_num = 200
# 평행이동 상수
shift_pixel = 315
#색상 범위 HSV
boundaries = [
    (np.array([161, 155, 84], dtype="uint8"), np.array([179, 255, 255], dtype="uint8")), # red1
    (np.array([0, 100, 70], dtype="uint8"), np.array([20, 255, 255], dtype="uint8")), # red2
    (np.array([94, 80, 200], dtype="uint8"), np.array([126, 255, 255], dtype="uint8")), # blue
    (np.array([0, 40, 60], dtype="uint8"), np.array([25, 255, 255], dtype="uint8")), #yellow
    (np.array([200, 0, 140], dtype="uint8"), np.array([255, 255, 255], dtype="uint8")) # white
]
# 모니터링 창 크기
display = (720, 480)
# 클로징 마스크 크기
kernel = np.ones((7,7), np.uint8)
#좌우 레인 디폴트 값
x1_default = int(0.37*display[0])
x2_default = int(0.63*display[0])

## Lane_Detection.py
class Lane_Detection: #Lane_Detction 클래스 생성후, original img 변경
    def __init__(self, img):  # 초기화
        self.x1_points1 = [None for _ in range(n_windows)]
        self.x1_points2 = [None for _ in range(n_windows)]
        self.x1_points3 = [None for _ in range(n_windows)]
        self.x2_points1 = [None for _ in range(n_windows)]
        self.x2_points2 = [None for _ in range(n_windows)]
        self.x2_points3 = [None for _ in range(n_windows)]
        self.original_img=img
        
        self.height, self.width = img.shape[:2]
        #roi설정을 위한 vertics, 위부터 차례대로 왼쪽 위, 왼쪽 아래, 오른쪽 아래, 오른쪽 위다.
        self.vertics = np.array([[(int(0.33*self.width), int(0.83*self.height)), 
                                  (int(0.16*self.width), int(self.height)),
                                  (int(0.84*self.width), int(self.height)),
                                  (int(0.67*self.width), int(0.83*self.height))]])
    
        #perspective변환을 위한 pts 설정
        self.pts1 = np.float32([(0.33*self.width, 0.83*self.height), 
                                (0.16*self.width, self.height),
                                (0.84*self.width, self.height),
                                (0.67*self.width, 0.83*self.height)])
        self.temp1, self.temp2 = display[:2]
        self.pts2 = np.float32([(0.33*self.temp1, 0*self.temp2),
                                (0.33*self.temp1, 1*self.temp2),
                                (0.66*self.temp1, 1*self.temp2),
                                (0.66*self.temp1, 0*self.temp2)])
        
        self.binary_img = self.make_binary()
        
        self.bin_height, self.bin_width = self.binary_img.shape[:2]
        cv2.imshow('bin', self.binary_img)
        print(self.bin_height,self.bin_width)
        self.interval_y = int(self.bin_height/(n_windows*2))
        self.interval_x = int(self.bin_width/windows_width)
        #왼쪽라인 보정상수
        self.line_constant1 = int(-(self.interval_x/0.7))
        #오른쪽라인 보정상수
        self.line_constant2 = int((self.interval_x/2))
        
        #조사창 y위치 정하기
        self.y_points = []
        y = self.interval_y
        for i in range (n_windows):
            self.y_points.append(y)
            y += self.interval_y*2
        if (self.y_points[-1]+self.interval_y) > self.bin_height :
            self.difference = self.y_points[-1] + self.interval_y-self.bin_height
            self.y_points[-1] -= self.difference
            
        self.left_points, self.right_points = self.make_route_points(self.binary_img)
        self.left_line(self.binary_img, self.left_points)
        self.right_line(self.binary_img, self.right_points)
        
    def left_line(self, img, left_points): #왼쪽 라인 추출(이미지로 리턴)
        img1 = np.zeros_like(img)
        self.draw_points(img1, left_points, self.y_points, [255, 0, 0], thickness =3)
        img1 = self.draw_lane(img1, self.left_points, self.y_points, [255, 0, 0])
        cv2.imshow('img',img1)
        return img1
        
    def right_line(self, img, right_points): #오른쪽 라인 추출(이미지로 리턴)
        img1 = np.zeros_like(img)
        self.draw_points(img1, right_points, self.y_points, [0, 255, 0], thickness =3)
        img1 = self.draw_lane(img1, self.right_points, self.y_points, [0, 255, 0])
        cv2.imshow('im2',img1)
        return img1
    
    def get_stop_line(self):  # 정지선을 반환하는 코드(정지선 제일 앞 부분)
        print(0)
        
    # 차선인식 알고리즘
    def make_route_points(self, img):
        h, w = img.shape[:2]
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
            for i in range(left_end-self.interval_x-1, self.interval_x-1, -1):
                count = 0
                minimg = img1[self.y_points[n]-self.interval_y: self.y_points[n]+self.interval_y, #작은 조사창 만들기
                           i-self.interval_x: i+self.interval_x]
                count = cv2.countNonZero(minimg)
                if(count>max_count):
                    max_count = count
                    max_point = i
                if(count>max_pixel_num):
                    break
            if max_count == 0 or max_count < min_pixel_num:
                x1_points.append(None)
            else:
                x1_points.append((max_point+int(self.interval_x/2)+self.line_constant1))

        # 오른쪽 라인 최대 x2 points 구하기
        x2_points=[]
        for n in range(0, n_windows):
            max_count = 0
            max_point = 0
            for i in range(self.interval_x, right_end-self.interval_x):
                count = 0
                minimg = img2[self.y_points[n]-self.interval_y: self.y_points[n]+self.interval_y,
                              i-self.interval_x: i+self.interval_x]
                count = cv2.countNonZero(minimg)
                if(count>max_count):
                    max_count = count
                    max_point = i
                if(count>max_pixel_num):
                    break
            if max_count == 0 or max_count < min_pixel_num:
                x2_points.append(None)
            else:
                x2_points.append(max_point+int(self.interval_x/2)+self.line_constant2)
                
        # x2 points에 왼쪽라인 이미지 길이를 더하여 오른쪽 라인에 피팅
        for i in range(n_windows):
            if(x2_points[i]!=None):
                x2_points[i]+=left_end

        #둘 중 하나가 None일 때 하나의 점을 평행이동 시켜 대입
        for i in range(n_windows):
            if(x1_points[i]==None and x2_points[i]!=None):
                x1_points[i]=x2_points[i]-shift_pixel
            elif(x1_points[i]!=None and x2_points[i]==None):
                x2_points[i]=x1_points[i]+shift_pixel
        
        self.x1_points3 = self.x1_points2
        self.x1_points2 = self.x1_points1
        self.x1_points1 = x1_points
        
        #이전 포인트들을 조사하여 None이 아닌 값들 평균
        for i in range(len(x1_points)):
            count = 0
            if(self.x1_points1[i] != None):
                    count += 1
            else:
                self.x1_points1[i] = x1_default
                x1_points[i] = self.x1_points1[i]
                count += 1
            if(self.x1_points2[i] != None):
                x1_points[i]+=self.x1_points2[i]
                count += 1
            if(self.x1_points3[i] != None):
                x1_points[i]+=self.x1_points3[i]
                count += 1
            if(count!=0):
                x1_points[i] = int(x1_points[i]/count)
                
        self.x2_points3 = self.x2_points2
        self.x2_points2 = self.x2_points1
        self.x2_points1 = x2_points
        for i in range(len(x2_points)):
            count = 0
            if(self.x2_points1[i] == None):
                x2_points[i]= x2_default
                count +=1
            else:
                count += 1
            if(self.x2_points2[i] != None):
                self.x2_points1[i]+=self.x2_points2[i]
                count += 1
            if(self.x2_points3[i] != None):
                self.x2_points1[i]+=self.x2_points3[i]
                count += 1
            if(count!=0):
                x2_points[i] = int(self.x2_points1[i]/count)
                
        return x1_points, x2_points
        
    '''
        for i in range(len(x1_points)-1):
            if(abs(x1_points[len(x1_points)-i-1] - x1_points[len(x1_points)-i-2])
               > del_cons):
                x1_points[len(x1_points)-i-2] = x1_points[len(x1_points)-i-1]
    
        for i in range(len(x2_points)-1):
            if(abs(x2_points[len(x2_points)-i-1] - x2_points[len(x2_points)-i-2])
               > del_cons):
                x2_points[len(x2_points)-i-2] = x2_points[len(x2_points)-i-1]
    '''
    def draw_points(self, img, x_points, y_points, color, thickness):
        try:
            for i in range(len(x_points)):
                if(x_points[i]!=None):
                    cv2.line(img, (int(x_points[i]), int(y_points[i])), (int(x_points[i]), int(y_points[i])),
                             color, thickness=9)
        except:
            print("error2")
            
    def draw_lane(self, img, x_points, y_points, color):
        line_x = []
        line_y = []
        for i in range(len(x_points)):
            if(x_points[i] != None):
                line_x.append(x_points[i])
                line_y.append(y_points[i])
        
        fit = np.polyfit(y_points, x_points, poly_order)
        line = np.poly1d(fit)
        x = np.linspace(0, display[0], display[0])
        y = np.round(line(x))
        
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        self.draw_points(img, y, x, color, thickness = 3)
    
        return img
    
    # 아래부터는 유틸함수
    def make_binary(self): # 이진화 이미지를 만드는 함수
        img = self.reg_of_int(self.original_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        warped_img = self.warp_image(img)
        img1 = np.zeros_like(warped_img)
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
        for index in ['b','y','w']: # 색깔별로 채널 추출
            img2 = self.detectcolor(warped_img, index)
            img1 = cv2.bitwise_or(img1, img2)
            
        img1 = self.closeimage(img1)
        return img1
        
    def reg_of_int(self, img): # 이미지에서 roi 잘라내기
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, self.vertics, (255,255,255))
        mask = cv2.bitwise_and(img, mask)
        return mask
        
    def warp_image(self, img): # 이미지 원근 변환
        M = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        warped_img = cv2.warpPerspective(img, M, display)
        return warped_img
    
    def detectcolor(self, img, color): # color = b, r, w, y / 이미지 상에서 색을 찾아 리턴
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
    
    def closeimage(self, img):
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
video="images_shor.jpg"
img = cv2.imread(video)
cv2.imshow('hi', img)
img1 = Lane_Detection(img)
cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()