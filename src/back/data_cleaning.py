import streamlit as st
import pandas as pd
import pandas as pd
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


def delete_rows_with_missing_data(df, threshold_percentage):
    """
        Delete rows in a DataFrame from a certain threshold

        Args:
        - df (pd.DataFrame): Dataframe to clean
        - threshold_percentage (int): Threshold selected by the user, if missing datas are greater than the
                                    threshold, then we would delete this row

        Returns:
        - modified_df: A new data frame
    """
    threshold = threshold_percentage / 100
    missing_data_per_row = df.isnull().sum(axis=1) / df.shape[1]
    mask = missing_data_per_row < threshold
    modified_df = df.loc[mask]
    return modified_df


def delete_cols_with_missing_data(df, threshold_percentage):
    """
        Delete columns in a DataFrame from a certain threshold

        Args:
        - df (pd.DataFrame): Dataframe to clean
        threshold_percentage (int): Threshold selected by the user, if missing datas are greater than the
                                    threshold, then we would delete this column

        Returns:
        - modified_df: A new data frame
    """
    threshold = threshold_percentage / 100
    missing_data_per_col = df.isnull().sum(axis=0) / df.shape[0]
    mask = missing_data_per_col < threshold
    modified_df = df.loc[:, mask]
    return modified_df


def replace_with_mean(df):
    """
        Replace values with the mean of the values

        Args:
        - df (pd.DataFrame): Dataframe to clean

        Returns:
        - The same data frame modified with the mean
    """
    return df.fillna(df.mean())


def replace_with_median(df):
    """
        Replace values with the median of the values

        Args:
        df (pd.DataFrame): Dataframe to clean

        Returns:
        - The same data frame modified
    """
    return df.fillna(df.median())


def replace_with_mode(df):
    """
        Replace values with the mode of the values

        Args:
        - df (pd.DataFrame): Dataframe to clean

        Returns:
        - The same data frame modified
    """
    for column in df.columns:
        mode_value = df[column].mode()
        if not mode_value.empty:
            df[column] = df[column].fillna(mode_value[0])
    return df


def knn_impute(df, n_neighbors):
    """
    Impute missing values in a DataFrame using KNN method

    Args:
    - df (pd.DataFrame): Dataframe to clean
    - n_neighbors (int): Number of neighbors to use for imputation

    Returns:
    - DataFrame with missing values imputed
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
    return imputed_df


def regression_impute(df, estimator, max_iter=10):
    """
    Impute missing values in a DataFrame using iterative regression

    Args:
    - df (pd.DataFrame): Dataframe to clean
    - estimator: Regressor to use for imputation. Default is BayesianRidge
    - max_iter (int): Maximum number of imputation iterations

    Returns:
    - DataFrame with missing values imputed
    """
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=0)
    imputed_array = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
    return imputed_df


def get_estimator(estimator_name):
    """
    Get the regression estimator based on the name in the choice list

    Args:
    - estimator_name: Name to parse

    Returns:
    - The right estimator, default BayesianRidge
    """
    estimators = {
        "BayesianRidge": BayesianRidge(),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR()
    }
    return estimators.get(estimator_name, BayesianRidge())


def normalize_min_max(df):
    """
    Normalize a DataFrame with the Min-Max method

    Args:
    - df: Dataframe to normalize

    Returns:
    - The dataframe normalized
    """
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized


def normalize_z_score(df):
    """
    Normalize a DataFrame with the z-score method

    Args:
    - df: Dataframe to normalize

    Returns:
    - The dataframe normalized
    """
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized


def remove_string_columns(df):
    """
    Remove string columns from a DataFrame
    Args:
    - df: DataFrame to clean

    Returns:
    - DataFrame without string columns
    """
    string_columns = df.select_dtypes(include=["object"]).columns
    df.drop(string_columns, axis=1, inplace=True)
    return df


def label_encode_strings(df):
    """
    Label encode string columns in a DataFrame

    Args:
    - df: DataFrame to clean

    Returns:
    - DataFrame with string columns label encoded

    """
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if not categorical_cols.empty:
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df


def one_hot_encode_strings(df):
    """
    One-hot encode string columns in a DataFrame

    Args:
    - df: DataFrame to clean

    Returns:
    - DataFrame with string columns one-hot encoded
    """
    string_columns = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=string_columns)
    return df


def do_remove_columns(df, columns):
    """
    Fonction qui supprime les colonnes spécifiées d'un DataFrame

    Args:
    - df (pd.DataFrame): Dataframe à nettoyer
    - columns (list): Liste des colonnes à supprimer

    Returns:
    - DataFrame nettoyé
    """
    if columns:
        df = df.drop(columns=columns)
    return df
