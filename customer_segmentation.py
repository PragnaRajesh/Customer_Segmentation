# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load and Preprocess the Data
# Load the dataset from an existing CSV file
data = pd.read_csv('customer_data.csv')  # Replace 'customer_data.csv' with the path to your dataset

# Check for missing values and handle them (example with imputation if necessary)
data.fillna(data.mean(), inplace=True)

# Normalize numerical columns using StandardScaler
scaler = StandardScaler()
data[['age', 'tenure', 'monthly_spending', 'num_products']] = scaler.fit_transform(data[['age', 'tenure', 'monthly_spending', 'num_products']])

# Visualize the distribution of features
features = ['age', 'tenure', 'monthly_spending', 'num_products']
for feature in features:
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Step 2: Hierarchical Clustering
# Selecting features for clustering
features_data = data[['age', 'tenure', 'monthly_spending']]

# Perform Hierarchical Clustering with AgglomerativeClustering
# We'll start with 3 clusters as an initial choice
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(features_data)
data['cluster'] = labels

# Step 3: Cluster Evaluation
# Plot a dendrogram to visualize the clustering process
linkage_matrix = linkage(features_data, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Dendrogram for Customer Segmentation')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Step 4: Cluster Profiling
# Calculate summary statistics for each cluster
cluster_summary = data.groupby('cluster')[['age', 'tenure', 'monthly_spending']].agg(['mean', 'median', 'std'])
print("Cluster Summary Statistics:")
print(cluster_summary)

# Visualize the clusters with pair plots
sns.pairplot(data, vars=['age', 'tenure', 'monthly_spending'], hue='cluster', palette='viridis')
plt.show()

# Save the processed data to a CSV file
data.to_csv('customer_segments.csv', index=False)
