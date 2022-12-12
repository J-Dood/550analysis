# Jordan Dood and Braeden Sopp
# 12/11/2022

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load the data from the csv file
data = pd.read_csv('polymerazeData2.csv')

# Extract the labels from the first column
labels = data.iloc[:, 0]

# Extract the features from the remaining columns
features = data.iloc[:, 1:]

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(features)

# Perform DBSCAN clustering
#dbscan = DBSCAN()
#dbscan.fit(features)

# Use PCA to reduce the number of features to 2 dimensions
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Print the formula for each component of the PCA
for component in pca.components_:
    print(" + ".join("{:.3f} * {}".format(value, name)
                     for value, name in zip(component, data.columns[1:])))

# Plot the data, colored by the predicted cluster
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.labels_)
plt.show()





