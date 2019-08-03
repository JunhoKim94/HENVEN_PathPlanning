import numpy as np
import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread('C:/photo/images/image_straight.jpg', cv2.IMREAD_GRAYSCALE)
#bilateral 필터 적용
blur2 = cv2.bilateralFilter(img1, 9, 75, 75)
ret, img = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

'''
cv2.calcHist(img, channel, mask, histSize, range)

이 함수는 이미지 히스토그램을 찾아서 numpy 배열로 리턴함
img: 히스토그램을 찾을 이미지. 인자는 반드시 []로 둘러싸야 함.
channel: grayscale 이미지의 경우 [0]을 인자로 입력하며,
컬러 이미지의 경우 B, G, R에 대한 히스토그램을 위해 각각 [0], [1], [2]를 인자로 입력
mask: 이미지 전체에 대한 히스토그램을 구할 경우 None,
이미지의 특정 영역에 대한 히스토그램을 구할 경우
이 영역에 해당하는 mask 값을 입력
histSize: BIN 개수. 인자는 []로 둘러싸야 함.
range: 픽셀값 범위. 보통 [0, 256]


즉, hist1은 grayscale 이미지인 img1 모든 영역에 대해
256개 BIN으로 0~255 픽셀값 범위의 히스토그램을 나타내는 numpy 배열입니다.
'''

#hist1 = cv2.calcHist([img], [0], None, [256], [0,256])
#plt.hist(img1.ravel(), 256, [0,256])
#plt.plot(hist1, color='256')
#plt.show()

h, w = img.shape[:2]
result=[]
result_x=[]
for i in range(w):
    result_x.append(i)
result_x2 = np.array(result_x)
px = img[100, 200]
print(px)
for i in range(0, w):
    cnt=0
    for j in range(0, h):
        px = img[j, i]
        if px != 0:
            cnt+=1
    result.append(cnt)
result2 = np.array(result)
print(result2)
plt.plot(result_x2, result2)
plt.show()
cv2.imshow('wow', img)
