import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import quick_data
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import os

GRAPH_PATH = "dbscan_graphs"
if not os.path.exists(GRAPH_PATH):
    os.makedirs(GRAPH_PATH)

def main():
    result = quick_data()

    # Select relevant columns for clustering
    results = result[["persid","agegroup_fine", "overall_trip_efficiency", "wasted_time", "persinc_fine", "totalwfh_ord","travel_time"]].dropna()
    KNN_df = result[["agegroup_fine", "overall_trip_efficiency", "wasted_time", "persinc_fine", "totalwfh_ord","travel_time"]].dropna()

    # Scales required Data to prevent bias
    scaler = MinMaxScaler()
    norm_KNN_data = pd.DataFrame(scaler.fit_transform(KNN_df), columns=KNN_df.columns, index=KNN_df.index)

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
    plt.savefig(os.path.join(GRAPH_PATH, "agg_clustering.png"))
    plt.show()

    # DBS SCAN clustering
    """AI declaration: The prompts on ChatGPT 5 'how to use DBSCAN clustering in python'
    and 'how to graph table of results of DBSCAN clustering' 
    were used in supporting the creation of the following code and graph."""
    db = DBSCAN(eps=0.08, min_samples=5)
    data_clean["db_cluster"] = db.fit_predict(data_clean[["pca1", "pca2"]])



    sns.scatterplot(data=data_clean, x="pca1", y="pca2", hue="db_cluster", palette="tab10", s=60)
    plt.title("DBSCAN: Automatic Detection of Natural Layers")
    plt.savefig(os.path.join(GRAPH_PATH, "dbscan_clustering.png"))
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
    plt.savefig(os.path.join(GRAPH_PATH, "mode_vs_dbscan_cluster.png"))
    plt.show()


    # Analyze feature similarities

    data_clean.drop(columns=["pca1", "pca2","most_used_mode", "agg_cluster"], inplace=True)
    db_cluster_summary = data_clean.groupby("db_cluster").mean(numeric_only=True)
    print("\nCluster feature means:\n", db_cluster_summary)

    # Visualize feature means per cluster
    plt.figure(figsize=(10, 6))
    sns.heatmap(db_cluster_summary, annot=True, cmap="coolwarm")
    plt.title("Average Feature Values per DBSCAN Cluster")
    plt.savefig(os.path.join(GRAPH_PATH, "dbscan_cluster_feature_means.png"))
    plt.show()

if __name__ == "__main__":
    print("Running DBSCAN.py")
    main()