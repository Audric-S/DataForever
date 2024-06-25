import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def data_clustering(df):
    st.title('Clustering')
    st.write("Appliquez PCA avant le clustering pour réduire la dimensionnalité des données.")

    apply_pca = st.checkbox('Appliquer PCA')
    if apply_pca:
        df = apply_pca_transform(df)

    algo = st.selectbox('Choisissez un algorithme de clustering', ('K-means', 'DBSCAN'))

    if algo == 'K-means':
        k_means_clustering(df)
    elif algo == 'DBSCAN':
        dbscan_clustering(df)

def k_means_clustering(df):
    st.subheader('Paramètres K-means')
    k = st.slider('Nombre de clusters (k)', 1, 10, 4)
    kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++').fit(df)
    df['Cluster'] = kmeans.labels_
    st.write('Clusters:', df['Cluster'].value_counts())
    st.write(df)

def dbscan_clustering(df):
    st.subheader('Paramètres DBSCAN')
    eps = st.slider('Epsilon (eps)', 0.1, 1.0, 0.5)
    min_samples = st.slider('Min_samples', 1, 10, 5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    df['Cluster'] = dbscan.labels_
    st.write('Clusters:', df['Cluster'].value_counts())
    st.write(df)

def data_prediction(df):
    st.title('Prédiction')
    st.write("Choisissez l'algorithme de prédiction et réglez ses paramètres pour voir les résultats.")

    target_columns = [col for col in df.columns if col != 'Cluster']
    target = st.selectbox('Choisissez la colonne cible', target_columns)
    X = df.drop(columns=[target])
    y = df[target]

    if pd.api.types.is_numeric_dtype(y):
        if pd.api.types.is_float_dtype(y):
            regression_prediction(X, y)
        elif pd.api.types.is_integer_dtype(y):
            classification_prediction(X, y)
    else:
        st.warning("Format de données non supporté, veuillez nettoyer vos données")

def regression_prediction(X, y):
    st.subheader('Régression')
    st.info("La colonne cible est continue. Nous effectuons une tâche de régression.")
    algo = st.selectbox('Choisissez un algorithme de régression', ('Linear Regression', 'Decision Tree Regressor'))

    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        st.warning("Les variables catégoriques détectées. Elles seront encodées pour la prédiction.")
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    if algo == 'Linear Regression':
        st.subheader('Paramètres Linear Regression')
        model = LinearRegression()
    elif algo == 'Decision Tree Regressor':
        st.subheader('Paramètres Decision Tree Regressor')
        max_depth = st.slider('Profondeur max', 1, 20, 5)
        model = DecisionTreeRegressor(max_depth=max_depth)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Erreur quadratique moyenne (MSE):", mean_squared_error(y_test, y_pred))

def classification_prediction(X, y):
    st.subheader('Classification')
    st.info("La colonne cible est catégorique. Nous effectuons une tâche de classification.")

    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        st.warning("Les variables catégoriques détectées. Elles seront encodées pour la prédiction.")
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    algo = st.selectbox('Choisissez un algorithme de classification', ('Logistic Regression', 'Decision Tree Classifier'))

    if algo == 'Logistic Regression':
        logistic_regression(X, y)
    elif algo == 'Decision Tree Classifier':
        decision_tree_classifier(X, y)

def logistic_regression(X, y):
    st.subheader('Paramètres Logistic Regression')
    max_iter = st.slider('Nombre max d’itérations', 100, 500, 200)
    model = LogisticRegression(max_iter=max_iter)
    display_classification_results(X, y, model)

def decision_tree_classifier(X, y):
    st.subheader('Paramètres Decision Tree Classifier')
    max_depth = st.slider('Profondeur max', 1, 20, 5)
    model = DecisionTreeClassifier(max_depth=max_depth)
    display_classification_results(X, y, model)

def display_classification_results(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Rapport de classification:")
    st.text(classification_report(y_test, y_pred))


def apply_pca_transform(df):
    st.subheader('Analyse en Composantes Principales (PCA)')
    st.write("Réglez le nombre de composantes principales pour effectuer l'analyse PCA.")

    n_components = st.slider('Nombre de composantes principales', 1, min(len(df.columns), 10), 2)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)

    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    st.write("Variance expliquée par chaque composante principale:", pca.explained_variance_ratio_)
    st.write(pca_df)
    
    coordvar = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=df.columns)

    fig, axes = plt.subplots(figsize=(6, 6))
    fig.suptitle("Cercle des corrélations")
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)

    axes.axvline(x=0, color='lightgray', linestyle='--', linewidth=1)
    axes.axhline(y=0, color='lightgray', linestyle='--', linewidth=1)

    for j in range(len(coordvar)):
        axes.text(coordvar.iloc[j, 0], coordvar.iloc[j, 1], coordvar.index[j])

    plt.gca().add_artist(plt.Circle((0, 0), 1, color='blue', fill=False))

    st.pyplot(fig)
    
    return pca_df


def main_prediction():
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choisissez une tâche", ("Prédiction", "Clustering"))

    if 'data_clean' not in st.session_state:
        st.session_state['data_clean'] = pd.DataFrame()

    df = st.session_state['data_clean']

    if df.empty:
        st.write("No data loaded yet. Please upload a CSV file first.")
    else:
        if option == "Clustering":
            data_clustering(df)
        elif option == "Prédiction":
            data_prediction(df)

if __name__ == "__main__":
    main_prediction()
