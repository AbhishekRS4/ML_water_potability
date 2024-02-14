import numpy as np
import pandas as pd


def read_csv_file(file_csv):
    """
    ---------
    Arguments
    ---------
    file_csv : str
        full path to dataset csv file

    -------
    Returns
    -------
    df_csv : pd.DataFrame
        a pandas dataframe for the csv file
    """
    df_csv = pd.read_csv(file_csv)
    return df_csv


def get_dict_nan_counts_per_col(data_frame):
    """
    ---------
    Arguments
    ---------
    data_frame : pd.DataFrame
        a pandas dataframe of some dataset

    -------
    Returns
    -------
    dict_nan_counts_per_col : dict
        a dictionary of NaN counts per column
    """
    dict_nan_counts_per_col = data_frame.isna().sum().to_dict()
    dict_nan_counts_per_col = dict(
        sorted(dict_nan_counts_per_col.items(), key=lambda kv: kv[1], reverse=True)
    )
    return dict_nan_counts_per_col


def get_data_from_data_frame(data_frame):
    """
    ---------
    Arguments
    ---------
    data_frame : pd.DataFrame
        a pandas dataframe of some dataset

    -------
    Returns
    -------
    (X_arr, Y_arr) : tuple
        a tuple of numpy arrays representing features and labels respsectively
    """
    arr = data_frame.to_numpy()
    X_arr, Y_arr = arr[:, :-1], arr[:, -1:].reshape(-1)
    return X_arr, Y_arr
