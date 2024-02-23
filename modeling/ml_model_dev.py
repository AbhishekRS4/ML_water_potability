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
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    KFold,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
)

from data_utils import WaterPotabilityDataLoader


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


class ClassificationPipeline:
    def __init__(self):
        self._imputer = None
        self._imputer_params = None
        self._preprocessor = None
        self._preprocessor_params = None
        self._transformer = None
        self._pca = None
        self._pca_params = None
        self._classifier = None
        self._classifier_params = None

        self.clf_pipeline = None
        self.clf_pipeline_params = None

    def set_imputer(self, imputer_type):
        """
        ---------
        Arguments
        ---------
        imputer_type : str
            a string indicating the type of imputer to be used in the ML pipeline
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

        self._imputer = imputer
        self._imputer_params = imputer_params
        return

    def set_preprocessor(self, preprocessor_type):
        """
        ---------
        Arguments
        ---------
        preprocessor_type : str
            a string indicating the type of preprocessor to be used in the ML pipeline
        """
        if preprocessor_type == "std":
            preprocessor = StandardScaler()
            preprocessor_params = None
        elif preprocessor_type == "min_max":
            # range must be positive for box-cox transformation
            # so use min max scaler and make sure range is proper
            # so use the same range even without that transformation
            preprocessor = MinMaxScaler(feature_range=(1, 2), clip=True)
            preprocessor_params = None
        elif preprocessor_type == "norm":
            preprocessor = Normalizer()
            preprocessor_params = {
                "preprocessing__norm": ["l1", "l2", "max"],
            }
        elif preprocessor_type == "poly":
            preprocessor = PolynomialFeatures()
            preprocessor_params = {
                "preprocessor__degree": [2],
                "preprocessor__interaction_only": [True, False],
                "preprocessor__include_bias": [True, False],
            }
        elif preprocessor_type == "robust":
            preprocessor = RobustScaler()
            preprocessor_params = None
        else:
            print(
                f"unidentified option for arg, preprocessor_type: {preprocessor_type}"
            )
            sys.exit(0)

        self._preprocessor = preprocessor
        self._preprocessor_params = preprocessor_params
        return

    def set_transformer(self, transformer_type):
        """
        ---------
        Arguments
        ---------
        transformer_type : str
            a string indicating the transformer type to be used in the ML pipeline
        """
        if transformer_type == "power_box_cox":
            self._transformer = PowerTransformer(method="box-cox")
        elif transformer_type == "power_yeo_johnson":
            self._transformer = PowerTransformer(method="yeo-johnson")
        else:
            print(
                f"unidentified option for arg, transformer_type: {transformer_type}"
            )
            sys.exit(0)
        return

    def set_pca(self, max_num_feats):
        """
        ---------
        Arguments
        ---------
        max_num_feats : int
            an integer indicating the maximum number of features in the dataset
        """
        self._pca = PCA()
        self._pca_params = {
            "pca__n_components": np.arange(2, max_num_feats + 1),
        }
        return

    def set_classifier(self, classifier_type):
        """
        ---------
        Arguments
        ---------
        classifier_type : str
            a string indicating the type of classifier to be used in the ML pipeline
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

        self._classifier = classifier
        self._classifier_params = classifier_params
        return

    def build_pipeline(self):
        if self._pca == None:
            if self._preprocessor == None:
                self.clf_pipeline = Pipeline(
                    [("imputer", self._imputer), ("classifier", self._classifier)]
                )
                list_pipeline_params = [self._imputer_params, self._classifier_params]
            else:
                if self._transformer == None:
                    self.clf_pipeline = Pipeline(
                        [
                            ("imputer", self._imputer),
                            ("preprocessor", self._preprocessor),
                            ("classifier", self._classifier),
                        ]
                    )
                else:
                    self.clf_pipeline = Pipeline(
                        [
                            ("imputer", self._imputer),
                            ("preprocessor", self._preprocessor),
                            ("transformer", self._transformer),
                            ("classifier", self._classifier),
                        ]
                    )
                if self._preprocessor_params is not None:
                    list_pipeline_params = [
                        self._imputer_params,
                        self._preprocessor_params,
                        self._classifier_params,
                    ]
                else:
                    list_pipeline_params = [
                        self._imputer_params,
                        self._classifier_params,
                    ]
        else:
            # Preprocessing is a must for applying PCA
            self.clf_pipeline = Pipeline(
                [
                    ("imputer", self._imputer),
                    ("preprocessor", self._preprocessor),
                    ("pca", self._pca),
                    ("classifier", self._classifier),
                ]
            )
            if self._preprocessor_params is not None:
                list_pipeline_params = [
                    self._imputer_params,
                    self._preprocessor_params,
                    self._pca_params,
                    self._classifier_params,
                ]
            else:
                list_pipeline_params = [
                    self._imputer_params,
                    self._pca_params,
                    self._classifier_params,
                ]
        self._set_pipeline_params(list_pipeline_params)
        return

    def _set_pipeline_params(self, list_pipeline_params):
        """
        ---------
        Arguments
        ---------
        list_pipeline_params : list
            a list of dictionaries of pipeline params
        """
        final_pipeline_params = {}
        for _index in range(len(list_pipeline_params)):
            temp_pipeline_params = {
                **final_pipeline_params,
                **list_pipeline_params[_index],
            }
            final_pipeline_params = temp_pipeline_params
        self.clf_pipeline_params = final_pipeline_params
        return


def train_model(
    water_pot_dataset, imputer_type, preprocessor_type, transformer_type, classifier_type, is_pca=False
):
    """
    ---------
    Arguments
    ---------
    water_pot_dataset : object
        an object of type WaterPotabilityDataLoader class
    imputer_type : str
        a string indicating the imputer type to be used in the ML pipeline
    preprocessor_type : str
        a string indicating the preprocessor type to be used in the ML pipeline
    transformer_type : str
        a string indicating the additional transformer type to be used in the ML pipeline
    classifier_type : str
        a string indicating the classifier type to be used in the ML pipeline
    is_pca : bool
        a boolean indicating whether to use PCA or not in the ML pipeline
    """
    # get data arrays from the data frame for train and test sets
    X_train, Y_train = water_pot_dataset.get_data_from_data_frame(which_set="train")
    X_test, Y_test = water_pot_dataset.get_data_from_data_frame(which_set="test")

    pca_str = "no_pca"
    preprocessor_str = "no_preproc"
    transformer_str = "no_transform"

    clf_pipeline = ClassificationPipeline()

    # set imputer and its params
    clf_pipeline.set_imputer(imputer_type)

    # set preprocessor and its params
    if preprocessor_type != "none":
        clf_pipeline.set_preprocessor(preprocessor_type)
        preprocessor_str = preprocessor_type

    # set the additional transformer if needed
    if transformer_type != "none":
        transformer_str = transformer_type
        if transformer_type == "power_box_cox":
            # range must be positive for box-cox transformation
            # so use min max scaler and make sure range is proper
            clf_pipeline.set_preprocessor("min_max")
            clf_pipeline.set_transformer(transformer_type)
            preprocessor_str = "min_max"
        else:
            # std scaler for yeo-johnson transformation yields better results
            clf_pipeline.set_preprocessor("std")
            clf_pipeline.set_transformer(transformer_type)
            preprocessor_str = "std"

    # to use PCA or not
    if is_pca == True:
        clf_pipeline.set_pca(X_train.shape[1])
        pca_str = "pca"

    # set classifier and its params
    clf_pipeline.set_classifier(classifier_type)

    print("\n" + "-" * 100)
    # build the model pipeline
    clf_pipeline.build_pipeline()
    print(clf_pipeline.clf_pipeline)

    print("\n" + "-" * 100)
    print("Model pipeline params space: ")
    print(clf_pipeline.clf_pipeline_params)
    print("-" * 100)

    # setup grid search with k-fold cross validation
    k_fold_cv = KFold(n_splits=5, shuffle=True, random_state=4)
    grid_cv = GridSearchCV(
        clf_pipeline.clf_pipeline,
        clf_pipeline.clf_pipeline_params,
        scoring="f1",
        cv=k_fold_cv,
    )
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
    model_log_str = f"{imputer_type}_{preprocessor_str}_{transformer_str}_{pca_str}_{classifier_type}"
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # log the model and the metrics
        mlflow.sklearn.log_model(cv_best_estimator, model_log_str)
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
    water_pot_dataset = WaterPotabilityDataLoader(ARGS.file_csv)
    water_pot_dataset.read_csv_file()
    water_pot_dataset.split_data()

    num_samples_train = water_pot_dataset.df_train.shape[0]
    num_samples_test = water_pot_dataset.df_test.shape[0]

    print("\n" + "-" * 40)
    print("Num samples after splitting the dataset")
    print("-" * 40)
    print(f"train: {num_samples_train}, test: {num_samples_test}")

    print("\n" + "-" * 40)
    print("A few samples from train data")
    print("-" * 40)
    print(water_pot_dataset.df_train.head())

    if ARGS.is_train:
        train_model(
            water_pot_dataset,
            ARGS.imputer_type,
            ARGS.preprocessor_type,
            ARGS.transformer_type,
            ARGS.classifier_type,
            bool(ARGS.is_pca),
        )
    return


def main():
    file_csv = "dataset/water_potability.csv"
    classifier_type = "ada_boost"
    imputer_type = "knn"
    preprocessor_type = "none"
    transformer_type = "none"
    is_train = 1
    is_pca = 0

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
        help="classifier to be used in the ML model pipeline",
    )
    parser.add_argument(
        "--imputer_type",
        default=imputer_type,
        type=str,
        choices=["simple", "knn", "iterative"],
        help="imputer to be used in the ML model pipeline",
    )
    parser.add_argument(
        "--preprocessor_type",
        default=preprocessor_type,
        type=str,
        choices=["none", "std", "min_max", "norm", "poly", "robust"],
        help="preprocessor to be used in the ML model pipeline",
    )
    parser.add_argument(
        "--transformer_type",
        default=transformer_type,
        type=str,
        choices=["none", "power_box_cox", "power_yeo_johnson"],
        help="additional transformer to be used in the ML model pipeline",
    )
    parser.add_argument(
        "--is_pca",
        default=is_pca,
        type=int,
        choices=[0, 1],
        help="indicates if pca should be used in the ML model pipeline (0: False, 1: True)",
    )

    ARGS, unparsed = parser.parse_known_args()
    init_and_train_model(ARGS)
    return


if __name__ == "__main__":
    main()
