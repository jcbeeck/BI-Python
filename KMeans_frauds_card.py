
"""
Jan Beeck

Fraud Detection Segemention
"""

# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sea
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# reading the data frame

#https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

data = pd.read_csv('card_transdata.csv')

#data.head()

#Feature Explanation:

#distance_from_home - the distance from home where the transaction happened.
#distance_from_last_transaction - the distance from last transaction happened.
#ratio_to_median_purchase_price - Ratio of purchased price transaction
#                                to median purchase price.
#repeat_retailer - Is the transaction happened from same retailer.
#used_chip - Is the transaction through chip (credit card).
#used_pin_number - Is the transaction happened by using PIN number.
#online_order - Is the transaction an online order.
#fraud - Is the transaction fraudulent.

#Strategy of indicators with physical cards
#repeat_retailer (ind4) used_chip (ind5) fraud (ind8)

d1 = data.iloc[:, 4].values #repeat_retailer
d2 = data.iloc[:, 5].values #used_chip
d3 = data.iloc[:, 7].values #fraud

# Creating the dataset's array or dataframe
dataset = np.column_stack((d1,d2,d3))


dataset = dataset[0:10000, 0:3]

dataset = pd.DataFrame(dataset)

# Creation of a model

kmeans = KMeans(n_clusters=5, random_state=0)

kmeans.fit(dataset)

print(silhouette_score(dataset, kmeans.labels_, metric='euclidean'))
#0.9619741063820227 (5 clusters)
#0.9996005763688761 (6 clusters)

clusters = kmeans.fit_predict(dataset.iloc[:,1:])

#-------------------------------------------------------------

#Visualization of the segmentation through the PCA method

reduced_data = PCA(n_components=2).fit_transform(dataset)
kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()












