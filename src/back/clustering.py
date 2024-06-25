import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def apply_pca_transform(df, n_components):
    pca = PCA(n_components=n_components)
    pca_df = pca.fit_transform(df)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=df.columns)
    return pca_df, pca.explained_variance_ratio_, loadings

def k_means_clustering(df, k):
    kmeans_pca = KMeans(n_clusters=k, n_init=10, init='k-means++')
    kmeans_pca.fit(df)
    cluster_labels = kmeans_pca.labels_
    centroids = kmeans_pca.cluster_centers_
    return df, cluster_labels, centroids

def dbscan_clustering(df, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    df['Cluster'] = dbscan.labels_
    return df

def visualize_clusters_2d(pca_result):
    st.subheader('Visualisation des clusters (2D)')
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], cmap='viridis')
    plt.title('KMeans Clustering on PCA Results')
    st.pyplot(plt)

def visualize_clusters_3d(pca_result, cluster_labels, centroids):
    cluster_labels_str = cluster_labels.astype(str)

    fig = px.scatter_3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        color=cluster_labels,
        labels={'color': 'Cluster'}
    )

    fig.add_trace(
        px.scatter_3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
    ).data[0])


    st.plotly_chart(fig)




