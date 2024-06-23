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
from back.data_cleaning_back import *



def main():
    st.title('Welcome on the data cleaning page')
    st.subheader('Data cleaning', divider='grey')

    select_remove_method()
    method = select_method()
    handle_method_selection(method)

    if st.button("Submit your cleaning"):
        if validate_inputs(method):
            if 'data' in st.session_state:
                df = st.session_state.get('data_clean', st.session_state.get('data'))
                df = perform_cleaning(df, method)
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
    df = st.session_state['data']
    st.write("Would you like to remove the data of type string ?")
    st.button("Yes")
    st.button("No")


def select_method():
    method = st.selectbox(
        "How would you like to deal with missing values?",
        ("Delete datas", "Datas replacement", "Sophisticated imputation"),
        index=None,
        placeholder="Select a method"
    )
    st.write("You selected:", method)
    return method


def handle_method_selection(method):
    if method == "Delete datas":
        handle_delete_datas()
    elif method == "Datas replacement":
        handle_datas_replacement()
    elif method == "Sophisticated imputation":
        handle_sophisticated_imputation()


def handle_delete_datas():
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
    normalizing = st.selectbox(
        "How would you like to normalize your data?",
        ("Min Max", "Z-score"),
        index=None,
        placeholder="Select a method"
    )
    return normalizing


def validate_inputs(method):
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
    if method is None or normalizing is None:
        st.error("Please select a valid option in all the select boxes.")
        return False
    return True


def perform_cleaning(df, method):
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
    if 'data' in st.session_state:
        df = st.session_state['data']
        main()
    else:
        st.write("No data loaded yet. Please upload a CSV file first.")


def perform_normalization(df, normalizing_method):
    if normalizing_method == "Min Max":
        df_normalized = normalize_min_max(df)
    elif normalizing_method == "Z-score":
        df_normalized = normalize_z_score(df)
    else:
        raise ValueError("Invalid normalization method selected.")

    st.session_state['data_clean'] = df_normalized
    return df_normalized
