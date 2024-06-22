import streamlit as st
import numpy as np
import plotly.express as px


# Fonction pour la visualisation des clusters en 2D
def visualize_clusters_2d(X):
    st.scatter_chart(X, size=len(X))


# Fonction pour la visualisation des clusters en 3D
def visualize_clusters_3d(X, labels, centers=None):
    fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels)
    st.plotly_chart(fig)


# Fonction principale de l'application Streamlit
def cluster_visualisation():
    st.title('Visualisation des Clusters')

    # Exemple de données et de labels (à remplacer par les vôtres)
    two_dimension_data = np.random.rand(100, 2)  # Exemple de données
    labels = np.random.randint(0, 3, size=100)  # Exemple de labels de clustering

    # Visualisation en 2D des clusters
    visualize_clusters_2d(two_dimension_data)

    # Exemple de données et de labels (à remplacer par les vôtres)
    three_dimension_data = np.random.rand(100, 3)  # Exemple de données

    # Visualisation en 3D des clusters
    visualize_clusters_3d(three_dimension_data, labels)
