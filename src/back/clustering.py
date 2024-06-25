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
    return df, cluster_labels

def dbscan_clustering(df, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    df['Cluster'] = dbscan.labels_
    return df

def visualize_clusters_2d(pca_result, cluster_labels):
    st.subheader('Visualisation des clusters (2D)')
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering on PCA Results')
    st.pyplot(plt)

def visualize_clusters_3d(pca_result, cluster_labels):
    st.subheader('Visualisation des clusters (3D)')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=cluster_labels, cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('3D KMeans Clustering on PCA Results')
    st.pyplot(plt)
