import json
import requests
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from modeling.ml_model_dev import read_csv_file


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def send_post_reqest(ARGS):
    df_csv = read_csv_file(ARGS.file_csv)

    df_train, df_test = train_test_split(df_csv, test_size=0.1, random_state=4)
    list_cols = df_train.columns[:-1]
    X_test, Y_test = df_test.to_numpy()[:, :-1], df_test.to_numpy()[:, -1:]
    print(X_test.shape)

    url = "http://0.0.0.0:5000/predict"
    # the endpoint of the post request

    headers = {"Content-type": "application/json"}
    # additional headers to indicate the content type of the post request

    # perform 20 post requests
    for i in range(0, 20):
        list_values = list(X_test[i, :])
        encoded_data = dict(zip(list_cols, list_values))
        print(encoded_data)
        result = requests.post(url, data=json.dumps(encoded_data), headers=headers)
        print(f"{json.loads(result.text)} \n")
        # print(f"{type(json.loads(result.text))} \n")
    return


def main():
    file_csv = "dataset/water_potability.csv"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--file_csv", default=file_csv, type=str, help="full path to dataset csv file"
    )

    ARGS, unparsed = parser.parse_known_args()
    send_post_reqest(ARGS)

    return


if __name__ == "__main__":
    main()
