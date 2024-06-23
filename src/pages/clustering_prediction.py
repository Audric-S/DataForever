import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def data_clustering(df):
    st.title('Clustering')
    st.write("Choisissez l'algorithme de clustering et réglez ses paramètres pour voir les résultats.")
    
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
        regression_prediction(X, y)
    else:
        classification_prediction(X, y)

def regression_prediction(X, y):
    st.subheader('Régression')
    st.info("La colonne cible est continue. Nous effectuons une tâche de régression.")
    algo = st.selectbox('Choisissez un algorithme de régression', ('Linear Regression', 'Decision Tree Regressor'))

    # Encodage des variables catégoriques
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

    if 'label_encoders' in locals():
        st.write("Colonnes encodées avec LabelEncoder:")
        for col, le in label_encoders.items():
            st.write(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
def linear_regression(X, y):
    st.subheader('Paramètres Linear Regression')
    model = LinearRegression()
    display_regression_results(X, y, model)

def decision_tree_regressor(X, y):
    st.subheader('Paramètres Decision Tree Regressor')
    max_depth = st.slider('Profondeur max', 1, 20, 5)
    model = DecisionTreeRegressor(max_depth=max_depth)
    display_regression_results(X, y, model)

def display_regression_results(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Erreur quadratique moyenne (MSE):", mean_squared_error(y_test, y_pred))

def classification_prediction(X, y):
    st.subheader('Classification')
    st.info("La colonne cible est catégorique. Nous effectuons une tâche de classification.")
    
    # Encodage des variables catégoriques
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

    if 'label_encoders' in locals():
        st.write("Colonnes encodées avec LabelEncoder:")
        for col, le in label_encoders.items():
            st.write(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

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

def main_prediction():
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choisissez une tâche", ("Clustering", "Prédiction"))

    if 'data' not in st.session_state:
        st.session_state['data'] = pd.DataFrame()

    df = st.session_state['data']
    
    if df.empty:
        st.write("No data loaded yet. Please upload a CSV file first.")
    else:
        if option == "Clustering":
            data_clustering(df)
        elif option == "Prédiction":
            data_prediction(df)

if __name__ == "__main__":
    main_prediction()
