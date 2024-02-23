import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class WaterPotabilityDataLoader:
    def __init__(self, file_csv, test_size=0.1, random_state=4):
        self.file_csv = file_csv
        self.test_size = test_size
        self.random_state = random_state

        self.df_csv = None
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def read_csv_file(self):
        self.df_csv = pd.read_csv(self.file_csv)
        return

    def split_data(self):
        self.df_train, self.df_test = train_test_split(
            self.df_csv, test_size=self.test_size, random_state=self.random_state
        )
        return

    def get_data_from_data_frame(self, which_set="train"):
        """
        ---------
        Arguments
        ---------
        which_set : str
            a string indicating for which set the data arrays should be returned

        -------
        Returns
        -------
        (X_arr, Y_arr) : tuple
            a tuple of numpy arrays of features and labels for the appropriate set
        """
        if which_set == "train":
            data_frame = self.df_train
        else:
            data_frame = self.df_test
        arr = data_frame.to_numpy()
        X_arr, Y_arr = arr[:, :-1], arr[:, -1:].reshape(-1)
        return X_arr, Y_arr


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
