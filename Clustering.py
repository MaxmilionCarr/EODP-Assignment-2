# question being:
# Which demographic feature are the most important in predicting transport mode and length 


from preprocessing import fetch_transport_mode
import clustering_process as cl_p
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
import seaborn as sns

import seaborn as sns

# DATA_TYPE_USED=["agegroup", "overall_trip_efficiency", "wasted_time", "distance", "persinc", "totalwfh","time"]
DATA_TYPE_USED=cl_p.PERSONS_COLUMNS+[ "overall_trip_efficiency", "wasted_time", "distance", "totalwfh","time"]


ANALYSIS_TYPES=['wasted_time','overall_trip_efficiency']
USED_ANALYSIS='persinc'
NUM_CLUSTER=10




def clustering():
    processed_data,first_indicators,maximums_scaling,mode_scaling =cl_p.processed_data_clustering(DATA_TYPE_USED)
    
    normalised_modes=processed_data[cl_p.ALLOWED_MODES+[cl_p.PER_WEIGHT]]
    

    
    processed_data.drop(columns=['persid']+cl_p.ALLOWED_MODES+[cl_p.PER_WEIGHT],axis=1, inplace=True)
    

    scaler = MinMaxScaler()
    norm_KNN_data = pd.DataFrame(scaler.fit_transform(processed_data.dropna()), columns=processed_data.columns) 
    processed_data = pd.concat([norm_KNN_data, normalised_modes], axis=1)

    weight=processed_data[cl_p.PER_WEIGHT]
    processed_data.drop(columns=[cl_p.PER_WEIGHT], axis=1, inplace=True)

    graph_efficiency(processed_data,weight)
    plot_kmeans(processed_data,weight)

    return 

def plot_kmeans(df,weight):
    data=df[[ANALYSIS_TYPES[0],ANALYSIS_TYPES[1]]]
    clusters = KMeans(n_clusters=NUM_CLUSTER,init='k-means++')
    

    clusters.fit(df,sample_weight=weight)
    colormap = {0: 'red', 1: 'green', 2: 'blue',3:'yellow',4:'purple',5:'pink',6:'orange',7:'brown',8:'cyan',9:'black'}

    fig = plt.figure(figsize=(7, 10))
    ax = plt.axes()
    ax.scatter(data[ANALYSIS_TYPES[1]], 
               data[ANALYSIS_TYPES[0]], 
               c=[colormap.get(x) for x in clusters.labels_])
    
    ax.set_ylabel(ANALYSIS_TYPES[0])
    ax.set_xlabel(ANALYSIS_TYPES[1])
    ax.set_title(f"k = {len(set(clusters.labels_))}")
    
    plt.show()

def graph_efficiency(df,weights):
    # count_k_nodes(df)
    colormap = {0: 'red', 1: 'green', 2: 'blue',3:'yellow',4:'purple',5:'pink',6:'orange',7:'brown',8:'cyan',9:'black'}
    column =df.columns
    
    sklearn_pca = PCA(n_components=3)
    X_pca = sklearn_pca.fit_transform(df)
    explained_var = sklearn_pca.explained_variance_ratio_

    print(f"Variance explained by each PC: {explained_var}")   
    print(f"Total variance explained: {explained_var.sum():.4f}")

    kmean = KMeans(n_clusters=NUM_CLUSTER,init='k-means++')

    clusters = kmean.fit(df,sample_weight=weights)
    sns.scatterplot(x=X_pca[:,0], 
                y=X_pca[:,1],
                hue=clusters.labels_,
                palette=colormap)
    plt.title("PCA with 2 components")
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.show()


    
    new_df=pd.DataFrame(kmean.cluster_centers_,columns=column)
    x=new_df[USED_ANALYSIS]
    y=new_df['overall_trip_efficiency']
    order = np.argsort(x)
    xs = np.array(x)[order]
    ys = np.array(y)[order]

    plt.plot(xs,ys)
    plt.ylabel('overall trip efficiency',fontsize=9)
    plt.xlabel(USED_ANALYSIS,fontsize=9)
    plt.title(USED_ANALYSIS+' vs overall trip efficiency',fontsize=7)
    plt.show()

    

    new_df = cl_p.most_often_mode(new_df)
    









def count_k_nodes(normalize_data):

    distortions = []
    k_range = range(1, 50)
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(normalize_data)
        distortions.append(kmeans.inertia_) # Question: What does kmeans.inertia_ return? 
        
    plt.plot(k_range, distortions, 'bx-')

    plt.title('The Elbow Method showing the optimal k')
    plt.xlabel('k')
    plt.ylabel('Distortion')

    plt.show()
clustering()

