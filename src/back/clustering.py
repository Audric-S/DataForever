import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler


def apply_pca_transform(df, n_components):
    """
    Fonction permettant d'appliquer une transformation PCA sur un DataFrame.

    Parameters:
    - df: DataFrame à transformer
    - n_components: nombre de composantes principales à garder

    Returns:
    - pca_df: DataFrame transformé
    - explained_variance_ratio: variance expliquée par chaque composante principale
    - loadings: DataFrame des vecteurs propres
    """
    pca = PCA(n_components=n_components)
    pca_df = pca.fit_transform(df)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(n_components)], index=df.columns)
    return pca_df, pca.explained_variance_ratio_, loadings


def k_means_clustering(df, k):
    """
    Fonction permettant de réaliser un clustering K-Means sur un DataFrame.

    Args:
    - df: DataFrame à clusteriser
    - k: nombre de clusters à créer

    Returns:
    - df: DataFrame avec les clusters
    - cluster_labels: labels des clusters
    - centroids: coordonnées des centroïdes
    """
    kmeans_pca = KMeans(n_clusters=k, n_init=10, init='k-means++')
    kmeans_pca.fit(df)
    cluster_labels = kmeans_pca.labels_
    centroids = kmeans_pca.cluster_centers_
    return df, cluster_labels, centroids


def dbscan_clustering(df, eps, min_samples):
    """
    Fonction permettant de réaliser un clustering DBSCAN sur un DataFrame.

    Args:
    - df: DataFrame à clusteriser
    - eps: distance maximale entre deux
    - min_samples: nombre minimal de points pour former un cluster

    Returns:
    - df_result: DataFrame avec les clusters
    - labels: labels des clusters
    - centroids: coordonnées des centroïdes
    """
    # scaler = StandardScaler()
    # df_scaled = scaler.fit_transform(df)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    labels = dbscan.labels_
    df_result = pd.DataFrame(df, columns=[f'PC{i + 1}' for i in range(df.shape[1])])
    df_result['Cluster'] = labels
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        if label == -1:
            continue
        centroid = np.mean(df_result[df_result['Cluster'] == label].iloc[:, :-1], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    return df_result, labels, centroids


def visualize_clusters_2d(pca_result, cluster_labels, centroids):
    """
    Fonction permettant de visualiser les clusters en 2D.

    Args:
    - pca_result: résultat de la PCA
    - cluster_labels: labels des clusters
    - centroids: coordonnées des centroïdes

    Returns:
    - None
    """
    st.subheader('Visualisation des clusters (2D)')
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', label='Data Points')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red', label='Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clustering on PCA Results')
    plt.legend()
    st.pyplot(plt)


def visualize_clusters_3d(pca_result, cluster_labels):
    """
    Fonction permettant de visualiser les clusters en 3D.

    Args:
    - pca_result: résultat de la PCA
    - cluster_labels: labels des clusters

    Returns:
    - None
    """
    st.subheader('Visualisation des clusters (3D)')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    mask = cluster_labels != -1
    pca_result_filtered = pca_result[mask]
    cluster_labels_filtered = cluster_labels[mask]

    if isinstance(pca_result_filtered, pd.DataFrame):
        ax.scatter(pca_result_filtered.iloc[:, 0], pca_result_filtered.iloc[:, 1], pca_result_filtered.iloc[:, 2],
                   c=cluster_labels_filtered, cmap='viridis')
    else:
        ax.scatter(pca_result_filtered[:, 0], pca_result_filtered[:, 1], pca_result_filtered[:, 2],
                   c=cluster_labels_filtered, cmap='viridis')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('3D DBSCAN Clustering on PCA Results')
    st.pyplot(plt)


def visualize_clusters_3d_interactive(pca_result, cluster_labels, centroids):
    """
    Fonction permettant de visualiser les clusters en 3D de manière interactive.

    Args:
    - pca_result: résultat de la PCA
    - cluster_labels: labels des clusters
    - centroids: coordonnées des centroïdes

    Returns:
    - None
    """
    mask = cluster_labels != -1
    pca_result_filtered = pca_result[mask]
    cluster_labels_filtered = cluster_labels[mask]

    cluster_labels_str = cluster_labels_filtered.astype(str)
    fig = px.scatter_3d(
        x=pca_result_filtered[:, 0],
        y=pca_result_filtered[:, 1],
        z=pca_result_filtered[:, 2],
        color=cluster_labels_str,
        labels={'color': 'Cluster'}
    )

    # Ajouter une trace séparée pour les centroides avec une couleur spécifique
    centroid_trace = go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color='black',  # Couleur des centroides
            symbol='diamond'
        ),
        name='Centroids'
    )

    # Ajouter la trace des centroides à la figure
    fig.add_trace(centroid_trace)

    # Afficher la figure avec Streamlit
    st.plotly_chart(fig)
