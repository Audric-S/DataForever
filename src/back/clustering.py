import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px

def apply_pca_transform(df, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    return pca_df, pca.explained_variance_ratio_

def k_means_clustering(df, k):
    kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++').fit(df)
    df['Cluster'] = kmeans.labels_
    return df

def dbscan_clustering(df, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    df['Cluster'] = dbscan.labels_
    return df

# Fonction pour la visualisation des clusters en 2D
def visualize_clusters_2d(dataframe):
    st.scatter_chart(dataframe, size=len(dataframe))


# Fonction pour la visualisation des clusters en 3D
def visualize_clusters_3d(dataframe):
    fig = px.scatter_3d(
        x=dataframe.iloc[:, 0],
        y=dataframe.iloc[:, 1],
        z=dataframe.iloc[:, 2]
    )
    st.plotly_chart(fig)
