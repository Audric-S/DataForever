import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

def data_clustering():
    st.title('Clustering')
    st.write("""
    Choisissez l'algorithme de clustering et réglez ses paramètres pour voir les résultats.
    """)

    if 'data' in st.session_state:
        data = st.session_state['data']
        chooseAlgorithm()
    else:
        st.write("No data loaded yet. Please upload a CSV file first.")


def chooseAlgorithm():
    algo = st.selectbox('Choisissez un algorithme de clustering', ('K-means', 'DBSCAN'))

    if algo == 'K-means':
        st.subheader('Paramètres K-means')
        k = st.slider('Nombre de clusters (k)', 1, 10, 4)
        kmeans = KMeans(n_clusters=k, n_init=1, init='k-means++').fit(data)
    elif algo == 'DBSCAN':
        st.subheader('Paramètres DBSCAN')
        eps = st.slider('Epsilon (eps)', 0.1, 1.0, 0.3)
        min_samples = st.slider('Min_samples', 1, 10, 5)


if __name__ == "__main__":
    data_clustering()
