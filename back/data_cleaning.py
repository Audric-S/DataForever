import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


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
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
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
def regression_impute(df, estimator=BayesianRidge(), max_iter=10):
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=0)
    imputed_array = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
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
