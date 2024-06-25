import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from back.prediction import perform_regression, perform_classification
import seaborn as sns
from back.clustering import apply_pca_transform, k_means_clustering, dbscan_clustering, visualize_clusters_2d, visualize_clusters_3d, visualize_clusters_3d_interactive
from sklearn.metrics import silhouette_score


def main_prediction_clustering():
    """
        Fonction principale pour l'interface de prédiction et clustering.
        Elle permet de choisir entre les deux tâches et de choisir les paramètres pour chaque tâche.

        Returns:
            - None
    """
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choisissez une tâche", ("Prédiction", "Clustering"))

    if 'data_clean' not in st.session_state:
        st.session_state['data_clean'] = pd.DataFrame()

    df = st.session_state['data_clean']

    if df.empty:
        st.write("No data loaded yet. Please upload a CSV file first.")
    else:
        if option == "Clustering":
            st.title('Welcome on the clustering page')
            n_components = st.slider('Nombre de composantes principales', 1, min(len(df.columns), 10), 2)
            pca_df, explained_variance_ratio, loadings = apply_pca_transform(df, n_components)

            st.subheader("Visualisation des loadings")
            fig, ax = plt.subplots()
            sns.heatmap(loadings, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            
            algo = st.selectbox('Choisissez un algorithme de clustering', ('K-means', 'DBSCAN'))
            if algo == 'K-means':
                k = st.slider('Nombre de clusters (k)', 1, 10, 4)
                standardize = st.checkbox('Standardiser les données (Standard Scaler)')
                if standardize:
                    scaler = StandardScaler()
                    pca_df_standardized = scaler.fit_transform(pca_df)
                    clustered_df, cluster_labels, centroids = k_means_clustering(pca_df_standardized, k)
                else:
                    clustered_df, cluster_labels, centroids = k_means_clustering(pca_df, k)
                
                silhouette_avg = silhouette_score(pca_df_standardized if standardize else pca_df, cluster_labels)
                st.write(f'Coefficient de silhouette pour {k} clusters: {silhouette_avg}')
                if n_components >= 3:
                    visualize_clusters_3d_interactive(clustered_df, cluster_labels, centroids)
                    visualize_clusters_3d(clustered_df, cluster_labels)
                if n_components >= 2:
                    visualize_clusters_2d(clustered_df, cluster_labels, centroids)
            elif algo == 'DBSCAN':
                eps = st.slider('Epsilon (eps)', 0.1, 1.0, 0.5)
                min_samples = st.slider('Min_samples', 1, 10, 5)
                standardize = st.checkbox('Standardiser les données (Standard Scaler)')
                if standardize:
                    scaler = StandardScaler()
                    pca_df_standardized = scaler.fit_transform(pca_df)
                    clustered_df, cluster_labels, centroids = dbscan_clustering(pca_df_standardized, eps, min_samples)
                else:
                    clustered_df, cluster_labels, centroids = dbscan_clustering(pca_df, eps, min_samples)
                
                st.write('Clusters:', clustered_df['Cluster'].value_counts())
                if len(set(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(pca_df_standardized if standardize else pca_df, cluster_labels)
                    st.write(f'Coefficient de silhouette pour les clusters trouvés: {silhouette_avg}')
                else:
                    st.write('Un seul cluster trouvé, le calcul du coefficient de silhouette n\'est pas pertinent.')
                
                if n_components >= 3:
                    visualize_clusters_3d_interactive(clustered_df.values, cluster_labels, centroids)
                    visualize_clusters_3d(clustered_df.values, cluster_labels)
                if n_components >= 2:
                    visualize_clusters_2d(clustered_df.values, cluster_labels, centroids)
        elif option == "Prédiction":
            st.title('Welcome on the prediction page')
            st.write("Choisissez l'algorithme de prédiction et réglez ses paramètres pour voir les résultats.")

            target_columns = [col for col in df.columns if col != 'Cluster']
            target = st.selectbox('Choisissez la colonne cible', target_columns)
            X = df.drop(columns=[target])
            y = df[target]

            if pd.api.types.is_numeric_dtype(y):
                if pd.api.types.is_float_dtype(y):
                    algo = st.selectbox('Choisissez un algorithme de régression', ('Linear Regression', 'Decision Tree Regressor'))
                    if algo == 'Linear Regression':
                        max_iter = st.slider('Nombre max d’itérations', 100, 10000, 1000, step=100)
                        mse, r2, mae, rmse = perform_regression(X, y, algo='Linear Regression', max_iter=max_iter)
                        st.write("Erreur quadratique moyenne (MSE):", mse)
                        st.write("Coefficient de détermination (R-squared):", r2)
                        st.write("Erreur absolue moyenne (MAE):", mae)
                        st.write("Erreur quadratique moyenne (RMSE):", rmse)
                    elif algo == 'Decision Tree Regressor':
                        max_depth = st.slider('Profondeur max', 1, 20, 5)
                        mse, r2, mae, rmse = perform_regression(X, y, algo='Decision Tree Regressor', max_depth=max_depth)
                        st.write("Erreur quadratique moyenne (MSE):", mse)
                        st.write("Coefficient de détermination (R-squared):", r2)
                        st.write("Erreur absolue moyenne (MAE):", mae)
                        st.write("Erreur quadratique moyenne (RMSE):", rmse)


                
                elif pd.api.types.is_integer_dtype(y):
                    algo = st.selectbox('Choisissez un algorithme de classification', ('Logistic Regression', 'Decision Tree Classifier'))
                    if algo == 'Logistic Regression':
                        max_iter = st.slider('Nombre max d’itérations', 100, 500, 200)
                        classification_report = perform_classification(X, y, algo='Logistic Regression', max_iter=max_iter)
                        st.write("Rapport de classification:")
                        st.text(classification_report)
                    elif algo == 'Decision Tree Classifier':
                        max_depth = st.slider('Profondeur max', 1, 20, 5)
                        classification_report = perform_classification(X, y, algo='Decision Tree Classifier', max_depth=max_depth)
                        st.write("Rapport de classification:")
                        st.text(classification_report)
            else:
                st.warning("Format de données non supporté, veuillez nettoyer vos données")

if __name__ == '__main__':
    main_prediction_clustering()
