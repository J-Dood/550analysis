# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data from csv file
data = pd.read_csv("data.csv")

# Perform clustering using KMeans and DBSCAN
kmeans = KMeans(n_clusters=3).fit(data)
dbscan = DBSCAN(eps=0.3, min_samples=10).fit(data)

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Run PCA
pca = PCA()
pca.fit(data_scaled)

# Plot biplot
biplot(data_scaled, pca)
plt.show()

# Plot scree plot
scree_plot(pca)
plt.show()

# Print formula for each component of PCA
for i, component in enumerate(pca.components_):
    print(f"PC{i+1} = {' + '.join(f'{value:.3f} * {data.columns[index]}' for index, value in enumerate(component))}")
