import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from back.data_cleaning import *



def main():
    """
        Fonction principale de la page de nettoyage des données qui permet de gérer les différentes méthodes de nettoyage.

        Returns:
            - None
    """
    st.title('Welcome on the data cleaning page')
    st.subheader('Data cleaning', divider='grey')

    remove_string_method = select_remove_method()
    method = select_method()
    handle_method_selection(method)

    if st.button("Submit your cleaning"):
        if validate_inputs(method):
            if 'data' in st.session_state:
                df = st.session_state.get('data_clean', st.session_state.get('data'))
                df = perform_cleaning(df, method, remove_string_method)
                st.session_state['data'] = df
                st.success("Data has been cleaned successfully!")
                st.dataframe(df)
            else:
                st.error("No data loaded yet. Please upload a CSV file first.")

    st.subheader('Data normalizing', divider='grey')
    normalizing = select_normalizing_method()

    st.write("You selected:", normalizing)

    if st.button("Submit your normalization"):
        if validate_normalization_inputs(method, normalizing):
            if 'data' in st.session_state:
                df = st.session_state.get(st.session_state['data_clean'], st.session_state['data'])
                df = perform_normalization(df, normalizing)
                st.session_state['data'] = df
                st.success("Data has been normalized successfully!")
                st.dataframe(df)
            else:
                st.error("No data loaded yet. Please upload a CSV file first.")

def select_remove_method():
    """
        Fonction qui permet de sélectionner la méthode de suppression des données.

        Returns:
            - remove_string_method (str): La méthode de suppression des données sélectionnée.
    """
    df = st.session_state['data']
    st.write("How would you like to handle string data?")
    remove_string_method = st.selectbox(
        "Select a method to handle string data:",
        ("Remove", "Label encoding", "One-hot encoding"),
        index=None,
        placeholder="Select a method"
    )
    st.write("You selected:", remove_string_method)
    return remove_string_method

def handle_remove_string(remove_string_method):
    """
        Fonction qui permet de gérer la méthode de suppression des données.

        Args:
            - remove_string_method (str): La méthode de suppression des données sélectionnée.

        Returns:
            - remove_strings (bool): True si la méthode de suppression est "Remove", False sinon.
            - label_encoding (bool): True si la méthode de suppression est "Label encoding", False sinon.
            - one_hot_encoding (bool): True si la méthode de suppression est "One-hot encoding", False sinon.
    """
    remove_strings = False
    label_encoding = False
    one_hot_encoding = False

    if remove_string_method == "Remove":
        remove_strings = True
    elif remove_string_method == "Label encoding":
        label_encoding = True
    elif remove_string_method == "One-hot encoding":
        one_hot_encoding = True

    return remove_strings, label_encoding, one_hot_encoding


def select_method():
    """
        Fonction qui permet de sélectionner la méthode de nettoyage des données.

        Returns:
            - method (str): La méthode de nettoyage des données sélectionnée.
    """
    method = st.selectbox(
        "How would you like to deal with missing values?",
        ("Delete datas", "Datas replacement", "Sophisticated imputation"),
        index=None,
        placeholder="Select a method"
    )
    st.write("You selected:", method)
    return method


def handle_method_selection(method):
    """
        Fonction qui permet de gérer la méthode de nettoyage des données.

        Args:
            - method (str): La méthode de nettoyage des données sélectionnée.

        Returns:
            - None
    """
    if method == "Delete datas":
        handle_delete_datas()
    elif method == "Datas replacement":
        handle_datas_replacement()
    elif method == "Sophisticated imputation":
        handle_sophisticated_imputation()


def handle_delete_datas():
    """
        Fonction qui permet de gérer la suppression des données.

        Returns:
            - None
    """
    delete_options = ("Rows", "Columns", "Both")
    delete_choice = st.selectbox(
        "What would you like to delete?",
        delete_options,
        index=None,
        placeholder="Select an option"
    )
    threshold = st.slider('Select the threshold percentage:', 0, 100, 50)
    st.write("You selected threshold:", threshold, "%")
    st.write("You selected to delete:", delete_choice)
    st.session_state.delete_choice = delete_choice
    st.session_state.threshold = threshold


def handle_datas_replacement():
    """
        Fonction qui permet de gérer le remplacement des données.

        Returns:
            - None
    """
    replacement_options = ("Mean", "Median", "Mode")
    replacement = st.selectbox(
        "How would you like to replace the missing values?",
        replacement_options,
        index=None,
        placeholder="Select a method"
    )
    st.write("You selected:", replacement)
    st.session_state.replacement = replacement


def handle_sophisticated_imputation():
    """
        Fonction qui permet de gérer l'imputation des données.

        Returns:
            - None
    """
    imputation_options = ("KNN", "Regression")
    imputation = st.selectbox(
        "How would you like to impute the missing values?",
        imputation_options,
        index=None,
        placeholder="Select a method"
    )
    st.write("You selected:", imputation)
    st.session_state.imputation = imputation

    if imputation == "KNN":
        num_neighbors = st.number_input("Choose the number of neighbors for KNN imputation:", value=1, min_value=1)
        st.write("You selected:", num_neighbors, "neighbors")
        st.session_state.num_neighbors = num_neighbors
    elif imputation == "Regression":
        regression_estimators = ("BayesianRidge", "LinearRegression", "Ridge", "RandomForestRegressor", "SVR")
        selected_estimator = st.selectbox(
            "Which regression estimator would you like to use?",
            regression_estimators,
            index=None,
            placeholder="Select an estimator"
        )
        st.write("You selected:", selected_estimator)
        st.session_state.selected_estimator = selected_estimator


def select_normalizing_method():
    """
        Fonction qui permet de sélectionner la méthode de normalisation des données.

        Returns:
            - normalizing (str): La méthode de normalisation des données sélectionnée.
    """
    normalizing = st.selectbox(
        "How would you like to normalize your data?",
        ("Min Max", "Z-score"),
        index=None,
        placeholder="Select a method"
    )
    return normalizing


def validate_inputs(method):
    """
        Fonction qui permet de valider les entrées de l'utilisateur.

        Args:
            - method (str): La méthode de nettoyage des données sélectionnée.

        Returns:
            - True si les entrées sont valides, False sinon.
    """
    if method == "Delete datas":
        if st.session_state.get('threshold') is None or st.session_state.get('delete_choice') is None:
            st.error("Please select a valid threshold and delete choice.")
            return False
    elif method == "Datas replacement":
        if st.session_state.get('replacement') is None:
            st.error("Please select a valid replacement method.")
            return False
    elif method == "Sophisticated imputation":
        if st.session_state.get('imputation') is None:
            st.error("Please select a valid imputation method.")
            return False
        elif st.session_state.get('imputation') == "KNN" and (
                st.session_state.get('num_neighbors') is None or st.session_state.get('num_neighbors') <= 0):
            st.error("Please select a valid number of neighbors for KNN.")
            return False
        elif st.session_state.get('imputation') == "Regression" and st.session_state.get('selected_estimator') is None:
            st.error("Please select a valid regression estimator.")
            return False
    return True


def validate_normalization_inputs(method, normalizing):
    """
        Fonction qui permet de valider les entrées de l'utilisateur pour la normalisation des données.

        Args:
            - method (str): La méthode de nettoyage des données sélectionnée.
            - normalizing (str): La méthode de normalisation des données sélectionnée.

        Returns:
            - True si les entrées sont valides, False sinon.
    """
    if method is None or normalizing is None:
        st.error("Please select a valid option in all the select boxes.")
        return False
    return True


def perform_cleaning(df, method, remove_string_method):
    """
        Fonction qui permet de nettoyer les données.

        Args:
            - df (pd.DataFrame): Le DataFrame des données.
            - method (str): La méthode de nettoyage des données sélectionnée.
            - remove_string_method (str): La méthode de suppression des données sélectionnée.

        Returns:
            - df (pd.DataFrame): Le DataFrame des données nettoyées.
    """
    remove_strings, label_encoding, one_hot_encoding = handle_remove_string(remove_string_method)

    if remove_strings:
        df = remove_string_columns(df)
    elif label_encoding:
        df = label_encode_strings(df)
    elif one_hot_encoding:
        df = one_hot_encode_strings(df)


    if method == "Delete datas":
        if st.session_state.delete_choice == "Rows":
            df = delete_rows_with_missing_data(df, st.session_state.threshold)
        elif st.session_state.delete_choice == "Columns":
            df = delete_cols_with_missing_data(df, st.session_state.threshold)
        elif st.session_state.delete_choice == "Both":
            df = delete_rows_with_missing_data(df, st.session_state.threshold)
            df = delete_cols_with_missing_data(df, st.session_state.threshold)
    elif method == "Datas replacement":
        if st.session_state.replacement == "Mean":
            df = replace_with_mean(df)
        elif st.session_state.replacement == "Median":
            df = replace_with_median(df)
        elif st.session_state.replacement == "Mode":
            df = replace_with_mode(df)
    elif method == "Sophisticated imputation":
        if st.session_state.imputation == "KNN":
            df = knn_impute(df, st.session_state.num_neighbors)
        elif st.session_state.imputation == "Regression":
            estimator = get_estimator(st.session_state.selected_estimator)
            df = regression_impute(df, estimator)

    st.session_state['data_clean'] = df
    return df


def data_cleaning():
    """
        Fonction qui permet de nettoyer les données.

        Returns:
            - None
    """
    if 'data' in st.session_state:
        df = st.session_state['data']
        main()
    else:
        st.write("No data loaded yet. Please upload a CSV file first.")


def perform_normalization(df, normalizing_method):
    """
        Fonction qui permet de normaliser les données.

        Args:
            - df (pd.DataFrame): Le DataFrame des données.
            - normalizing_method (str): La méthode de normalisation des données sélectionnée.

        Returns:
            - df_normalized (pd.DataFrame): Le DataFrame des données normalisées.
    """
    if normalizing_method == "Min Max":
        df_normalized = normalize_min_max(df)
    elif normalizing_method == "Z-score":
        df_normalized = normalize_z_score(df)
    else:
        raise ValueError("Invalid normalization method selected.")

    st.session_state['data_clean'] = df_normalized
    return df_normalized
