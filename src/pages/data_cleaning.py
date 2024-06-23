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
    st.session_state['data'] = df.select_dtypes(include=[np.number])


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


######################################################


"""
    Delete rows in a DataFrame from a certain threshold

    Parameters:
    df (pd.DataFrame): Dataframe to clean
    threshold_percentage (int): Threshold selected by the user, if missing datas are greater than the 
                                threshold, then we would delete this row

    Returns:
    A new data frame
"""
def delete_rows_with_missing_data(df, threshold_percentage):
    threshold = threshold_percentage / 100
    missing_data_per_row = df.isnull().sum(axis=1) / df.shape[1]
    mask = missing_data_per_row < threshold
    modified_df = df.loc[mask]
    return modified_df


"""
    Delete columns in a DataFrame from a certain threshold

    Parameters:
    df (pd.DataFrame): Dataframe to clean
    threshold_percentage (int): Threshold selected by the user, if missing datas are greater than the 
                                threshold, then we would delete this column

    Returns:
    A new data frame
"""
def delete_cols_with_missing_data(df, threshold_percentage):
    threshold = threshold_percentage / 100
    missing_data_per_col = df.isnull().sum(axis=0) / df.shape[0]
    mask = missing_data_per_col < threshold
    modified_df = df.loc[:, mask]
    return modified_df


"""
    Replace values with the mean of the values

    Parameters:
    df (pd.DataFrame): Dataframe to clean
   
    Returns:
    The same data frame modified
"""
def replace_with_mean(df):
    return df.fillna(df.mean())


"""
    Replace values with the median of the values

    Parameters:
    df (pd.DataFrame): Dataframe to clean
   
    Returns:
    The same data frame modified
"""
def replace_with_median(df):
    return df.fillna(df.median())


"""
    Replace values with the mode of the values

    Parameters:
    df (pd.DataFrame): Dataframe to clean
   
    Returns:
    The same data frame modified
"""
def replace_with_mode(df):
    for column in df.columns:
        mode_value = df[column].mode()
        if not mode_value.empty:
            df[column] = df[column].fillna(mode_value[0])
    return df


"""
Impute missing values in a DataFrame using KNN method

Parameters:
df (pd.DataFrame): Dataframe to clean
n_neighbors (int): Number of neighbors to use for imputation

Returns:
DataFrame with missing values imputed
"""
def knn_impute(df, n_neighbors):
    non_empty_df = df.dropna(axis=1, how='all')
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(non_empty_df)
    imputed_df = pd.DataFrame(imputed_array, columns=non_empty_df.columns)
    
    return imputed_df

"""
Impute missing values in a DataFrame using iterative regression

Parameters:
df (pd.DataFrame): Dataframe to clean
estimator: Regressor to use for imputation. Default is BayesianRidge
max_iter (int): Maximum number of imputation iterations

Returns:
DataFrame with missing values imputed
"""
def regression_impute(df, estimator, max_iter=10):
    non_empty_df = df.dropna(axis=1, how='all')
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=0)
    imputed_array = imputer.fit_transform(numeric_df)
    imputed_numeric_df = pd.DataFrame(imputed_array, columns=numeric_df.columns)
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    imputed_df = pd.concat([imputed_numeric_df, non_numeric_df], axis=1)
    imputed_df = imputed_df[df.columns]
    
    return imputed_df


"""
Get the regression estimator based on the name in the choice list

Parameters:
estimator_name: Name to parse

Returns:
The right estimator, default BayesianRidge
"""
def get_estimator(estimator_name):
    estimators = {
        "BayesianRidge": BayesianRidge(),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR()
    }
    return estimators.get(estimator_name, BayesianRidge())


"""
Normalize the data frame with the min-max method

Parameters:
df: DataFrame to normalize

Returns:
A normalized dataframe
"""
def normalize_min_max(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

def normalize_z_score(df):
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

def perform_normalization(df, normalizing_method):
    if normalizing_method == "Min Max":
        df_normalized = normalize_min_max(df)
    elif normalizing_method == "Z-score":
        df_normalized = normalize_z_score(df)
    else:
        raise ValueError("Invalid normalization method selected.")

    st.session_state['data_clean'] = df_normalized
    return df_normalized
