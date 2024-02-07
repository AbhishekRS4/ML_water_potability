import argparse
from sklearn.metrics import classification_report

from ml_model_dev import load_ml_model, train_test_split, read_csv_file

def test_ml_pipeline(ARGS):
    df_csv = read_csv_file(ARGS.file_csv)
    df_train, df_test = train_test_split(df_csv, test_size=0.1, random_state=4)
    arr_test = df_test.to_numpy()
    X_test, Y_test = arr_test[:, :-1], arr_test[:, -1:].reshape(-1)

    model_pipeline = load_ml_model(ARGS.pkl_file_name)
    Y_pred_test = model_pipeline.predict(X_test)
    print(classification_report(Y_test, Y_pred_test))
    return

def main():
    file_csv = "dataset/water_potability.csv"
    pkl_file_name = "trained_models/random_forest.pkl"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file_csv", default=file_csv,
        type=str, help="full path to dataset csv file")
    parser.add_argument("--pkl_file_name", default=pkl_file_name,
        type=str, help="full path to ml model pkl file")

    ARGS, unparsed = parser.parse_known_args()
    test_ml_pipeline(ARGS)
    return

if __name__ == "__main__":
    main()
