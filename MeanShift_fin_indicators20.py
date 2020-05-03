#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Web April 29 2020

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

#Import MeanShift from the ML Library 
from sklearn.cluster import MeanShift


# Loading the Data 
data = pd.read_excel('fin_indicators20.xlsx') 
print data.head() 

d1 = data.iloc[:, 1].values #indicator_1
d2 = data.iloc[:, 2].values #indicator_2
d3 = data.iloc[:, 3].values #indicator_3
d4 = data.iloc[:, 4].values #indicator_4
d5 = data.iloc[:, 5].values #indicator_5
d6 = data.iloc[:, 6].values #indicator_6
d7 = data.iloc[:, 7].values #indicator_7
d8 = data.iloc[:, 8].values #indicator_8

# Creating the dataset's array
dataset = np.column_stack((d1,d2,d3,d4,d5,d6,d7,d8))

# Train the ML algorithm
ms = MeanShift()
ms.fit(dataset)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters = len(np.unique(labels))

print("Generated Clusters:", n_clusters)

colors = 10*['r.','b.','c.','k.','y.','m.']

#print len(labels)

#print len(dataset)

for i in range(len(labels)):
   plt.plot(dataset[i][0], dataset[i][1],
            colors[labels[i]], markersize = 10)

#Print the countries with their 8 financial indicator and its label
for i in range(len(labels)):
  print("ind_1:",dataset[i][0],"ind_2:",dataset[i][1],
       "ind_3:",dataset[i][2],"ind_4:",dataset[i][3],
        "ind_5:",dataset[i][4],"ind_6:",dataset[i][5],
        "ind_7:",dataset[i][6],"ind_8:",dataset[i][7],
        "label:",labels[i])

