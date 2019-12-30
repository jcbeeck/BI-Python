#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:33:15 2019

@author: J. Beeck
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from sklearn.cluster import KMeans

x = [1,1,1,5,6,7,1,8,4,6,8,5,8,3,1,9,7,1,6,5,8,6,10,6,10,5,8,1,7,10]
y = [4174,6545,7094,7657,3624,3158,4691,6958,5223,7759,7170,3922,4340
     ,3666,4945,4124,7566,4133,3461,5344,4654,5275,4956,7472,3302,7414
,4686,4071,4256,7809]

plt.scatter(x,y)
plt.show()

X = np.array([[20,24,4174,1],
	      [10, 18, 4100,4],
              [5, 6, 4206,7],
              [7, 22, 5660,4],
              [20, 22, 5656,8],
              [6, 40, 6021,3],
              [18, 7, 6916,8],
              [10, 37, 3433,4],
              [6, 18, 6932,2],
              [5, 26, 3738,7],
              [2, 20, 7761,10],
              [17, 8, 6141,2],
              [12, 12, 6604,9],
              [14, 33, 7633,3],
              [3, 40, 4812,1],
              [16, 27, 5514,5],
              [4, 31, 5593,2],
              [1, 26, 3003,4],
              [9, 25, 7447,6],
              [8, 22, 3395,7],
              [13, 17, 7375,5],
              [10, 5, 4178,5],
              [7, 30, 5962,7],
              [13, 12, 5298,6],
              [9, 17, 4634,4],
              [16, 9, 5519,6],
              [11, 9, 7645,5],
              [5, 9, 7645,5],
              [7, 15, 3520,4],
              [8, 9, 6095,5]])

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.", "r.", "c."]

for i in range(len(X)):
    print("coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", s =150, linewidths = 5, zorder = 10)

plt.show()




