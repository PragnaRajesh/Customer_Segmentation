# Customer Segmentation using Hierarchical Clustering

## Overview
This project demonstrates how to perform customer segmentation using hierarchical clustering. It uses a sample customer dataset, preprocesses the data, and applies Agglomerative Clustering to segment customers based on features like `age`, `tenure`, `monthly_spending`, and `num_products`. The clusters are evaluated, and their characteristics are analyzed through summary statistics and visualizations.

## Project Structure
The repository contains the following files:
- `customer_data.csv`: A sample dataset containing customer information.
- `customer_clustering.py`: The Python script that performs the data preprocessing, clustering, and evaluation.
- `customer_segments.csv`: The output file containing the customer data with cluster labels after segmentation.

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - scipy

You can install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
Steps
1. Load and Preprocess the Data
The dataset is loaded, missing values are handled, and numerical features are normalized using StandardScaler to bring all features to a common scale.

2. Hierarchical Clustering
We apply Agglomerative Clustering to the dataset using selected features (age, tenure, monthly_spending). Initially, the number of clusters is set to 3, but you can modify this based on your evaluation.

3. Cluster Evaluation
A dendrogram is plotted to visualize the clustering process. The optimal number of clusters can be determined from the dendrogram.

4. Cluster Profiling
Summary statistics (mean, median, std) for each cluster are calculated. The clusters are then visualized using pair plots to understand their distribution across different features.

5. Output
The final clustered data is saved as customer_segments.csv, which contains the original customer data along with an additional cluster column that shows the assigned cluster for each customer.

Usage
Ensure that you have the customer_data.csv file in your project directory.
Run the Python script customer_clustering.py:
bash
Copy code
python customer_clustering.py

This will:
Perform hierarchical clustering.
Generate visualizations (histograms, dendrogram, pair plots).
Save the clustered data to customer_segments.csv.
Example Output
After running the script, you will receive the following:

A dendrogram visualizing the clustering process.
Summary statistics of each cluster (mean, median, and standard deviation of the features).
Pair plots to show the distribution of the clusters across features.
A CSV file (customer_segments.csv) containing the original data with an additional cluster column.
