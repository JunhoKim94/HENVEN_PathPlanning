# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans,DBSCAN
from scipy.cluster.hierarchy import fcluster
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=20, random_state=4)


lidar = np.ones(180)
angle = np.arange(-90,90)*np.pi/180


'''
convert r,theta to x,y coordinate
'''

lidar[20:70] = 0.4
lidar[120:180] =0.45

print(np.shape(lidar))

x = np.sin(angle)*lidar
y = np.cos(angle)*lidar

a = np.vstack([x,y])
#a = a.reshape(180,2)
'''
#k-means clustering
model = KMeans(n_clusters=4,init="k-means++")
model.fit(a)
predict = model.predict(a)
c = model.cluster_centers_
for i in range(180):
    if (predict[i] == 1):
        plt.scatter(x[i],y[i],c = 'r')
    elif (predict[i] == 0):
        plt.scatter(x[i],y[i],c = 'g')
    else:
        plt.scatter(x[i],y[i],c = 'b')
plt.show()

##
c = c.reshape(2,2)
a = c[0]
b = c[1]
plt.scatter(c[0],c[1],c='r')
plt.scatter(x,y)
plt.show()

a = np.vstack([x,y])
a = a.reshape(180,2)

#dbscan clustering
model2 = DBSCAN(eps=0.2,min_samples=10)
c = model2.fit_predict(a)
for i in range(180):
    if c[i] == 1:
        plt.scatter(x[i],y[i],c='red')
    elif c[i] == 0:
        plt.scatter(x[i],y[i],c='g')
    else:
        plt.scatter(x[i],y[i],c='black')
plt.show()
'''
f = []
f1 = dict()
j=0
for i in range(179):
    if abs(lidar[i]-lidar[i+1])<0.2:
        f.append(j) 
    else:
        f.append(j)
        j = j+1
f.append(j)
f = np.array([f])
apend = np.concatenate((a,f))
for i in range(180):
    if (f[0,i] == 1):
        plt.scatter(x[i],y[i],c = 'r')
    elif (f[0,i] == 0):
        plt.scatter(x[i],y[i],c = 'g')
    elif (f[0,i] == 2):
        plt.scatter(x[i],y[i],c = 'b')
    else:
        plt.scatter(x[i],y[i],c = 'y')
plt.show()