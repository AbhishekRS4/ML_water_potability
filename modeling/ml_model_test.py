import argparse

import mlflow
import numpy as np
from sklearn.metrics import classification_report

from data_utils import WaterPotabilityDataLoader
from ml_model_dev import load_mlflow_model


def test_ml_pipeline(ARGS):
    water_pot_dataset = WaterPotabilityDataLoader(ARGS.file_csv)
    water_pot_dataset.read_csv_file()
    water_pot_dataset.split_data()
    X_test, Y_test = water_pot_dataset.get_data_from_data_frame(which_set="test")

    model_pipeline = load_mlflow_model(ARGS.dir_mlflow_model)
    Y_pred_test = model_pipeline.predict(X_test)
    print(classification_report(Y_test, Y_pred_test))
    return


def main():
    file_csv = "dataset/water_potability.csv"
    dir_mlflow_model = "model_for_production"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--file_csv", default=file_csv, type=str, help="full path to dataset csv file"
    )
    parser.add_argument(
        "--dir_mlflow_model",
        default=dir_mlflow_model,
        type=str,
        help="full path to directory containing mlflow model",
    )

    ARGS, unparsed = parser.parse_known_args()
    test_ml_pipeline(ARGS)
    return


if __name__ == "__main__":
    main()
