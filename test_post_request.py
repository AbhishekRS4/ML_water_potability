import json
import requests
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from modeling.data_utils import WaterPotabilityDataLoader


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def send_post_reqest(ARGS):
    water_pot_dataset = WaterPotabilityDataLoader(ARGS.file_csv)
    water_pot_dataset.read_csv_file()
    water_pot_dataset.split_data()
    list_cols = water_pot_dataset.df_csv.columns[:-1]
    X_test, Y_test = water_pot_dataset.get_data_from_data_frame(which_set="test")
    print(X_test.shape)

    url = "http://0.0.0.0:5000/predict"
    # the endpoint of the post request

    headers = {"Content-type": "application/json"}
    # additional headers to indicate the content type of the post request

    # perform 20 post requests
    for i in range(0, ARGS.num_requests):
        list_values = list(X_test[i, :])
        encoded_data = dict(zip(list_cols, list_values))
        print(encoded_data)
        result = requests.post(url, data=json.dumps(encoded_data), headers=headers)
        print(f"{json.loads(result.text)} \n")
        # print(f"{type(json.loads(result.text))} \n")
    return


def main():
    file_csv = "dataset/water_potability.csv"
    num_requests = 20

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--file_csv", default=file_csv, type=str, help="full path to dataset csv file"
    )
    parser.add_argument(
        "--num_requests",
        default=num_requests,
        type=int,
        help="number of post requests to send",
    )

    ARGS, unparsed = parser.parse_known_args()
    send_post_reqest(ARGS)

    return


if __name__ == "__main__":
    main()
