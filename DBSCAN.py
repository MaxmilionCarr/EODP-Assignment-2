print("Running LMI.py")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from preprocessing import quick_data
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


# Finds set data needed for PCA preprocessing and clustering
EDUCATION_COL = ["persid", "journey_travel_time" ,"journey_distance","journey_elapsed_time"]
WORK_COL = ["persid", "journey_travel_time" ,"journey_distance","journey_elapsed_time"]
STOPS_COL = ["persid","travtime", "vistadist" , "duration"]



# Load the dataset
work_trip = pd.read_csv("datasets/journey_education.csv", usecols=WORK_COL)
education_trip = pd.read_csv("datasets/journey_work.csv", usecols=EDUCATION_COL)
stops = pd.read_csv("datasets/stops.csv", usecols=STOPS_COL)

# initialises data with preprocessing functions
result = quick_data()

#Compute travel efficiency as the average of (distance / travel time) across all trips for each individual (persid).

work_trip["wasted_time_work"] = pd.to_numeric(work_trip["journey_elapsed_time"], errors='coerce') - pd.to_numeric(work_trip["journey_travel_time"], errors='coerce')
work_trip["travel_time_work"] = work_trip["journey_travel_time"]
overall_work_time_waste = work_trip[["persid", "wasted_time_work","travel_time_work"]].drop_duplicates()

education_trip["wasted_time_education"] = pd.to_numeric(education_trip["journey_elapsed_time"], errors='coerce') - pd.to_numeric(education_trip["journey_travel_time"], errors='coerce')
education_trip["travel_time_education"] = education_trip["journey_travel_time"]
overall_education_time_waste = education_trip[["persid", "wasted_time_education","travel_time_education"]].drop_duplicates()

merged = pd.merge(overall_education_time_waste, overall_work_time_waste, on="persid", how="outer")

# Create binning categories for modes of transport
mapping = {
    "Bicycle": "Active", "Mobility Scooter": "Active", "Motorcycle": "Private",
    "Public Bus": "Public", "Rideshare Service": "Public", "School Bus": "Public",
    "Taxi": "Private", "Train": "Public", "Tram": "Public",
    "Vehicle Driver": "Private", "Vehicle Passenger": "Private", "Walking": "Active", "Other": "Private",
    "Plane" : "Public", "Running/jogging" : "Active"
}

#Maps binning categories
result["most_used_mode"] = result["most_used_mode"].replace(mapping)

# Pick work if it exists, otherwise education
merged["wasted_time"] = merged["wasted_time_work"].combine_first(merged["wasted_time_education"])
merged["travel_time"] = merged["travel_time_work"].combine_first(merged["travel_time_education"])

# Final tidy dataframe
overall_time_waste = merged[["persid", "wasted_time","travel_time"]]

# Merge all data COMPLETE PREPROCESSING
result = result.merge(overall_time_waste, on='persid', how='left')


# Select relevant columns for clustering
results = result[["persid","agegroup", "overall_trip_efficiency", "wasted_time", "persinc", "totalwfh","travel_time"]].dropna()
KNN_df = result[["agegroup", "overall_trip_efficiency", "wasted_time", "persinc", "totalwfh","travel_time"]].dropna()

# Scales required Data to prevent bias
scaler = MinMaxScaler()
norm_KNN_data = pd.DataFrame(scaler.fit_transform(KNN_df), columns=KNN_df.columns)

# Use normalised data (remove NaNs)
data_clean = norm_KNN_data




# Agglomerative clustering setup

# Use clean data
data_clean = norm_KNN_data.dropna().copy()

# Perform agglomerative clustering into 3 groups
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
data_clean["agg_cluster"] = agg.fit_predict(data_clean)

# Reduce to 2D for visualization

pca = PCA(n_components=2)
reduced = pca.fit_transform(data_clean.drop(columns=["agg_cluster"]))
data_clean["pca1"] = reduced[:, 0]
data_clean["pca2"] = reduced[:, 1]


# Plot PCA clusters
"""AI declaration: The prompt on ChatGPT 5 'how to make graph of pca components'
was used in supporting the creation of the following graph."""
plt.figure(figsize=(10,6))
sns.scatterplot(data=data_clean, x="pca1", y="pca2", hue="agg_cluster", palette="Set2", s=60)
plt.title("Agglomerative Clustering (3 Groups, Ward's Minimum Distance)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()

# DBS SCAN clustering
"""AI declaration: The prompts on ChatGPT 5 'how to use DBSCAN clustering in python'
 and 'how to graph table of results of DBSCAN clustering' 
 were used in supporting the creation of the following code and graph."""
db = DBSCAN(eps=0.08, min_samples=5)
data_clean["db_cluster"] = db.fit_predict(data_clean[["pca1", "pca2"]])



sns.scatterplot(data=data_clean, x="pca1", y="pca2", hue="db_cluster", palette="tab10", s=60)
plt.title("DBSCAN: Automatic Detection of Natural Layers")
plt.show()


# Create cross-tab (matrix)
data_clean["most_used_mode"] = result.loc[data_clean.index, "most_used_mode"]
data_clean = data_clean[data_clean["db_cluster"].isin([0, 1, 2])]
mode_cluster_matrix = pd.crosstab(data_clean["most_used_mode"], data_clean["db_cluster"])

print("\nMode vs. DBSCAN Cluster Matrix:\n")
print(mode_cluster_matrix)

# Visualize as heatmap
plt.figure(figsize=(8,5))
sns.heatmap(mode_cluster_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Travel Mode vs DBSCAN Cluster Matrix")
plt.xlabel("DBSCAN Cluster")
plt.ylabel("Travel Mode Category")
plt.show()


# Analyze feature similarities

data_clean.drop(columns=["pca1", "pca2","most_used_mode", "agg_cluster","agg3_layer"], inplace=True)
db_cluster_summary = data_clean.groupby("db_cluster").mean(numeric_only=True)
print("\nCluster feature means:\n", db_cluster_summary)

# Visualize feature means per cluster
plt.figure(figsize=(10, 6))
sns.heatmap(db_cluster_summary, annot=True, cmap="coolwarm")
plt.title("Average Feature Values per DFBSCAN Cluster")
plt.show()

