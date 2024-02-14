import os
import sys
import joblib
import argparse
import collections

import mlflow
import numpy as np

import lightgbm as lgbm

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import (
    cross_validate,
    train_test_split,
    GridSearchCV,
    KFold,
)

from data_utils import read_csv_file, get_data_from_data_frame


def load_mlflow_model(dir_mlflow_model):
    """
    ---------
    Arguments
    ---------
    dir_mlflow_model : str
        full direcotry path of the mlflow model

    -------
    Returns
    -------
    model_pipeline : object
        an object of the mlflow sklearn model pipeline
    """
    model_pipeline = mlflow.sklearn.load_model(dir_mlflow_model)
    return model_pipeline


def get_imputer(imputer_type):
    """
    ---------
    Arguments
    ---------
    imputer_type : str
        a string indicating the type of imputer to be used in the pipeline

    -------
    Returns
    -------
    (imputer, imputer_params) : tuple
        a tuple of imputer object and a dictionary of the imputer params
    """

    # setup parameter search space for different imputers
    imputer, imputer_params = None, None
    if imputer_type == "simple":
        imputer = SimpleImputer()
        imputer_params = {
            "imputer__strategy": ["mean", "median", "most_frequent"],
        }
    elif imputer_type == "knn":
        imputer = KNNImputer()
        imputer_params = {
            "imputer__n_neighbors": [5, 7],
            "imputer__weights": ["uniform", "distance"],
        }
    elif imputer_type == "iterative":
        imputer = IterativeImputer()
        imputer_params = {
            "imputer__initial_strategy": ["mean", "median", "most_frequent"],
            "imputer__imputation_order": ["ascending", "descending"],
        }
    else:
        print(f"unidentified option for arg, imputer_type: {imputer_type}")
        sys.exit(0)
    return imputer, imputer_params


def get_scaler():
    """
    -------
    Returns
    -------
    scaler : object
        a scaler object of type StandardScaler
    """
    scaler = StandardScaler()
    return scaler


def get_pca(max_num_feats):
    """
    ---------
    Arguments
    ---------
    max_num_feats : int
        an integer indicating the maximum number of features in the dataset

    -------
    Returns
    -------
    (pca, pca_params) : tuple
        a tuple of pca object and a dictionary of the pca params
    """
    pca = PCA()
    pca_params = {
        "pca__n_components": np.arange(2, max_num_feats + 1),
    }
    return pca, pca_params


def get_classifier(classifier_type):
    """
    ---------
    Arguments
    ---------
    classifier_type : str
        a string indicating the type of classifier to be used in the pipeline

    -------
    Returns
    -------
    (classifier, classifier_params) : tuple
        a tuple of classifier object and a dictionary of the classifier params
    """

    # setup parameter search space for different classifiers
    classifier, classifier_params = None, None
    if classifier_type == "ada_boost":
        classifier = AdaBoostClassifier()
        classifier_params = {
            "classifier__learning_rate": [0.5, 1, 1.5, 2, 2.5, 3],
            "classifier__n_estimators": [100, 200, 500],
        }
    elif classifier_type == "log_reg":
        classifier = LogisticRegression(max_iter=200, solver="saga")
        classifier_params = {
            "classifier__penalty": ["l1", "l2", "elasticnet"],
            "classifier__class_weight": [None, "balanced"],
            "classifier__C": [0.1, 0.5, 1, 2],
            "classifier__l1_ratio": np.arange(0.1, 1, 0.1),
        }
    elif classifier_type == "random_forest":
        classifier = RandomForestClassifier()
        classifier_params = {
            "classifier__n_estimators": [100, 250],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 10, 25, 50, 75],
            "classifier__min_samples_leaf": [1, 5, 10, 20],
            "classifier__min_samples_split": [2, 3, 4, 5],
        }
    elif classifier_type == "svc":
        classifier = SVC()
        classifier_params = {
            "classifier__C": [0.5, 1, 1.5, 2, 2.5],
            "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "classifier__degree": [2, 3, 4],
        }
    elif classifier_type == "light_gbm":
        classifier = lgbm.LGBMClassifier(
            boosting_type="gbdt", objective="binary", metric="auc", verbosity=-1
        )
        classifier_params = {
            "classifier__num_leaves": [15, 31, 63, 127, 255],
            "classifier__learning_rate": [0.1, 0.5, 1, 2],
            "classifier__n_estimators": [100, 500, 1000],
            "classifier__reg_lambda": [0.1, 0.5, 1],
            "classifier__min_data_in_leaf": [10, 20, 30, 50],
        }
    else:
        print(f"unidentified option for arg, classifier_type: {classifier_type}")
        sys.exit(0)

    return classifier, classifier_params


def get_pipeline_params(dict_A_params, dict_B_params):
    """
    ---------
    Arguments
    ---------
    dict_A_params : dict
        a dictionary of params
    dict_B_params : dict
        another dictionary of params

    -------
    Returns
    -------
    pipeline_params : dict
        a merged dictionary of the two input dictionaries
    """
    pipeline_params = {**dict_A_params, **dict_B_params}
    return pipeline_params


def train_model(df_train, df_test, imputer_type, classifier_type):
    """
    ---------
    Arguments
    ---------
    df_train : pd.DataFrame
        pandas dataframe for train set
    df_train : pd.DataFrame
        pandas dataframe for test set
    imputer_type : str
        a string indicating the imputer type to be used in the pipeline
    classifier_type : str
        a string indicating the classifier type to be used in the pipeline
    """
    # get data arrays from the data frame for train and test sets
    X_train, Y_train = get_data_from_data_frame(df_train)
    X_test, Y_test = get_data_from_data_frame(df_test)

    # get imputer and its params
    imputer, imputer_params = get_imputer(imputer_type)

    # get classifier and its params
    classifier, classifier_params = get_classifier(classifier_type)

    # get the pipeline params
    pipeline_params = get_pipeline_params(imputer_params, classifier_params)

    print("\n" + "-" * 100)
    # build the model pipeline
    if classifier_type == "svc" or classifier_type == "log_reg":
        scaler = get_scaler()
        pca, pca_params = get_pca(X_train.shape[1])
        print(
            f"Started training the model with the imputer: {imputer_type}, preprocessing: std_scaler + pca, classifier: {classifier_type}"
        )

        pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("scaler", scaler),
                ("pca", pca),
                ("classifier", classifier),
            ]
        )
        pipeline_params = get_pipeline_params(pipeline_params, pca_params)
    else:
        print(
            f"Started training the model with the imputer: {imputer_type}, classifier: {classifier_type}"
        )
        pipeline = Pipeline([("imputer", imputer), ("classifier", classifier)])
    print("Model pipeline params space: ")
    print(pipeline_params)
    print("-" * 100)

    # setup grid search with k-fold cross validation
    k_fold_cv = KFold(n_splits=5, shuffle=True, random_state=4)
    grid_cv = GridSearchCV(pipeline, pipeline_params, scoring="f1", cv=k_fold_cv)
    grid_cv.fit(X_train, Y_train)

    # get the cross validation score and the params for the best estimator
    cv_best_estimator = grid_cv.best_estimator_
    cv_best_f1 = grid_cv.best_score_
    cv_best_params = grid_cv.best_params_

    # predict and compute train set metrics
    Y_train_pred = cv_best_estimator.predict(X_train)
    train_f1 = f1_score(Y_train, Y_train_pred)
    train_acc = accuracy_score(Y_train, Y_train_pred)

    # predict and compute test set metrics
    Y_test_pred = cv_best_estimator.predict(X_test)
    test_f1 = f1_score(Y_test, Y_test_pred)
    test_acc = accuracy_score(Y_test, Y_test_pred)

    print("\n" + "-" * 50)
    # begin mlflow logging for the best estimator
    mlflow.set_experiment("water_potability")
    experiment = mlflow.get_experiment_by_name("water_potability")
    print(f"Started mlflow logging for the best estimator")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # log the model and the metrics
        mlflow.sklearn.log_model(cv_best_estimator, f"{imputer_type}_{classifier_type}")
        mlflow.sklearn.save_model(
            cv_best_estimator, f"{imputer_type}_{classifier_type}"
        )
        mlflow.log_params(cv_best_params)
        mlflow.log_metric("cv_f1_score", cv_best_f1)
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("train_acc_score", train_acc)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_acc_score", test_acc)
    # end mlflow logging
    mlflow.end_run()
    print(f"Completed mlflow logging for the best estimator")
    print("-" * 50)
    return


def init_and_train_model(ARGS):
    df_csv = read_csv_file(ARGS.file_csv)
    df_train, df_test = train_test_split(df_csv, test_size=0.1, random_state=4)

    num_samples_train = df_train.shape[0]
    num_samples_test = df_test.shape[0]

    print("\n" + "-" * 40)
    print("Num samples after splitting the dataset")
    print("-" * 40)
    print(f"train: {num_samples_train}, test: {num_samples_test}")

    print("\n" + "-" * 40)
    print("A few samples from train data")
    print("-" * 40)
    print(df_train.head())

    if ARGS.is_train:
        train_model(df_train, df_test, ARGS.imputer_type, ARGS.classifier_type)
    return


def main():
    file_csv = "dataset/water_potability.csv"
    classifier_type = "ada_boost"
    imputer_type = "knn"
    is_train = 1

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--file_csv", default=file_csv, type=str, help="full path to dataset csv file"
    )
    parser.add_argument(
        "--is_train", default=is_train, type=int, choices=[0, 1], help="to train or not"
    )
    parser.add_argument(
        "--classifier_type",
        default=classifier_type,
        type=str,
        choices=["ada_boost", "log_reg", "random_forest", "svc", "light_gbm"],
        help="classifier to be used in the training model pipeline",
    )
    parser.add_argument(
        "--imputer_type",
        default=imputer_type,
        type=str,
        choices=["simple", "knn", "iterative"],
        help="imputer to be used in the training model pipeline",
    )

    ARGS, unparsed = parser.parse_known_args()
    init_and_train_model(ARGS)
    return


if __name__ == "__main__":
    main()
