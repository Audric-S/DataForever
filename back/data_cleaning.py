import pandas as pd
import numpy as np

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
