import streamlit as st
import pandas as pd
from back.clustering import apply_pca_transform, k_means_clustering, dbscan_clustering, visualize_clusters_2d, visualize_clusters_3d
from back.prediction import perform_regression, perform_classification

def main_prediction_clustering():
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
            apply_pca = st.checkbox('Appliquer PCA')
            if apply_pca:
                n_components = st.slider('Nombre de composantes principales', 1, min(len(df.columns), 10), 2)
                pca_df, explained_variance_ratio = apply_pca_transform(df, n_components)
                st.write("Variance expliquée par chaque composante principale:", explained_variance_ratio)
                st.write(pca_df)

            algo = st.selectbox('Choisissez un algorithme de clustering', ('K-means', 'DBSCAN'))
            if algo == 'K-means':
                k = st.slider('Nombre de clusters (k)', 1, 10, 4)
                clustered_df = k_means_clustering(df, k)
                st.write('Clusters:', clustered_df['Cluster'].value_counts())
                st.write(clustered_df)
                visualize_clusters_2d(clustered_df)
                visualize_clusters_3d(clustered_df)
            elif algo == 'DBSCAN':
                eps = st.slider('Epsilon (eps)', 0.1, 1.0, 0.5)
                min_samples = st.slider('Min_samples', 1, 10, 5)
                clustered_df = dbscan_clustering(df, eps, min_samples)
                st.write('Clusters:', clustered_df['Cluster'].value_counts())
                st.write(clustered_df)
                visualize_clusters_2d(clustered_df)
                visualize_clusters_3d(clustered_df)

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
                        max_iter = st.slider('Nombre max d’itérations', 100, 500, 200)
                        mse = perform_regression(X, y, algo='Linear Regression', max_iter=max_iter)
                        st.write("Erreur quadratique moyenne (MSE):", mse)
                    elif algo == 'Decision Tree Regressor':
                        max_depth = st.slider('Profondeur max', 1, 20, 5)
                        mse = perform_regression(X, y, algo='Decision Tree Regressor', max_depth=max_depth)
                        st.write("Erreur quadratique moyenne (MSE):", mse)
                
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

