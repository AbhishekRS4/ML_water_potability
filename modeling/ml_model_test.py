import argparse

import mlflow
import numpy as np
from sklearn.metrics import classification_report

from ml_model_dev import load_mlflow_model, train_test_split, read_csv_file

def test_ml_pipeline(ARGS):
    df_csv = read_csv_file(ARGS.file_csv)
    df_train, df_test = train_test_split(df_csv, test_size=0.1, random_state=4)
    arr_test = df_test.to_numpy()
    X_test, Y_test = arr_test[:, :-1], arr_test[:, -1:].reshape(-1)

    model_pipeline = load_mlflow_model(ARGS.dir_mlflow_model)
    Y_pred_test = model_pipeline.predict(X_test)
    print(classification_report(Y_test, Y_pred_test))
    return

def main():
    file_csv = "dataset/water_potability.csv"
    dir_mlflow_model = "trained_models/knn_ada_boost"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_csv", default=file_csv,
        type=str, help="full path to dataset csv file")
    parser.add_argument("--dir_mlflow_model", default=dir_mlflow_model,
        type=str, help="full path to directory containing mlflow model")

    ARGS, unparsed = parser.parse_known_args()
    test_ml_pipeline(ARGS)
    return

if __name__ == "__main__":
    main()
