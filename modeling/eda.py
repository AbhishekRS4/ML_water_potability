import os
import sys
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data_utils import read_csv_file, get_data_from_data_frame


def do_eda(ARGS):
    data_frame = read_csv_file(ARGS.file_csv)
    label_counts = dict(data_frame[ARGS.target_column].value_counts())
    # print(label_counts)

    # plot a histogram
    plt.figure(figsize=(12, 12))
    plt.bar([str(l) for l in label_counts.keys()], label_counts.values(), width=0.5)
    plt.xlabel(f"{ARGS.target_column}", fontsize=20)
    plt.ylabel("Number of samples", fontsize=20)
    plt.title("Distribution of samples in the dataset", fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    """
    feat_cols = data_frame.columns[:-1]
    num_feat_cols = len(feat_cols)

    fig, axs = plt.subplots(num_feat_cols)
    fig.suptitle("Distribution of features")
    #axs.set_xlabel(ARGS.target_column)

    for col_index in range(num_feat_cols):
        column = feat_cols[col_index]
        not_nan_indices = list(data_frame[column].notna())
        lbl_with_not_nans = data_frame[ARGS.target_column][not_nan_indices]
        col_with_not_nans = data_frame[column][not_nan_indices]
        print(column, len(lbl_with_not_nans), len(col_with_not_nans))

        axs[col_index].scatter(lbl_with_not_nans, col_with_not_nans)
        axs[col_index].set(ylabel=column)
    plt.show()
    """

    plt.figure()
    corr_mat = data_frame.corr()
    sns.heatmap(corr_mat)
    plt.title("Feature correlation matrix", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    return


def main():
    file_csv = "dataset/water_potability.csv"
    target_column = "Potability"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file_csv", default=file_csv, type=str, help="full path to dataset csv file"
    )
    parser.add_argument(
        "--target_column",
        default=target_column,
        type=str,
        help="target label for which the EDA needs to be done",
    )
    ARGS, unparsed = parser.parse_known_args()
    do_eda(ARGS)
    return


if __name__ == "__main__":
    main()
