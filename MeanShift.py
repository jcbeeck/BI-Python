#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:55:22 2020

@author: J. Beeck

Mean Shift is a hierarchical clustering algorithm. 
In contrast to supervised machine learning,
clustering attempts to group data without labeling, 
the latter is very useful in the context of
Business Intelligence for study which cluster belongs to
 a specific set of business's indicators.
As opposed to the KMeans, when using MeanShift, 
you don't need to know the numbers of clusters (categories)
 beforehand.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift


# Loading the Data 
data = pd.read_excel('dataset.xlsx') 
print data.head() 

print len(data.values)

x = data.iloc[:, 4].values  #indicator_4
y = data.iloc[:, 1].values  #indicator_1

plt.scatter(x,y)
plt.show()

d1 = data.iloc[:, 1].values #indicator_1
d2 = data.iloc[:, 2].values #indicator_2
d3 = data.iloc[:, 3].values #indicator_3
d4 = data.iloc[:, 4].values #indicator_4

# Creating the dataset's array
dataset = np.column_stack((d1,d2,d3,d4))

print dataset

print (dataset.shape)

# Train the ML algorithm
ms = MeanShift()
ms.fit(dataset)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters = len(np.unique(labels))

print("Generated Clusters:", n_clusters)

colors = 10*['r.','b.','c.','k.','y.','m.']

print len(labels)

print len(dataset)

for i in range(len(labels)):
    plt.plot(dataset[i][0], dataset[i][1], colors[labels[i]], markersize = 10)
    
    print("indicator_1:",dataset[i][0],"ind_2:",dataset[i][1],
          "ind_3:",dataset[i][2],"ind_4:",dataset[i][3],
          "label:",labels[i])

