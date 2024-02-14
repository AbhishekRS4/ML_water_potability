import os
import logging
from typing import Union

import mlflow
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from config import settings

try:
    path_mlflow_model = "./model_for_production/"
    sklearn_pipeline = mlflow.sklearn.load_model(path_mlflow_model)
except:
    path_mlflow_model = "/data/model_for_production/"
    sklearn_pipeline = mlflow.sklearn.load_model(path_mlflow_model)

app = FastAPI()
logging.basicConfig(level=logging.INFO)


class WaterPotabilityDataItem(BaseModel):
    ph: Union[float, None] = np.nan
    Hardness: Union[float, None] = np.nan
    Solids: Union[float, None] = np.nan
    Chloramines: Union[float, None] = np.nan
    Sulfate: Union[float, None] = np.nan
    Conductivity: Union[float, None] = np.nan
    Organic_carbon: Union[float, None] = np.nan
    Trihalomethanes: Union[float, None] = np.nan
    Turbidity: Union[float, None] = np.nan


def predict_pipeline(data_sample):
    """
    ---------
    Arguments
    ---------
    data_sample : np.array
        a numpy array of shape (num_samples, num_feats)

    -------
    Returns
    -------
    pred_sample : np.array
        a numpy array of shape (num_samples) with predictions
    """
    pred_sample = sklearn_pipeline.predict(data_sample)
    return pred_sample


@app.get("/info")
def get_app_info():
    """
    -------
    Returns
    -------
    dict_info : dict
        a dictionary with info to be sent as a response to get request
    """
    dict_info = {"app_name": settings.app_name, "version": settings.version}
    return dict_info


@app.post("/predict")
def predict(wpd_item: WaterPotabilityDataItem):
    """
    ---------
    Arguments
    ---------
    wpd_item : object
        an object of type WaterPotabilityDataItem

    -------
    Returns
    -------
    pred_dict : dict
        a dictionary of prediction to be sent as a response to post request
    """
    wpd_arr = np.array(
        [
            wpd_item.ph,
            wpd_item.Hardness,
            wpd_item.Solids,
            wpd_item.Chloramines,
            wpd_item.Sulfate,
            wpd_item.Conductivity,
            wpd_item.Organic_carbon,
            wpd_item.Trihalomethanes,
            wpd_item.Turbidity,
        ]
    ).reshape(1, -1)
    logging.info("data sample: %s", wpd_arr)
    pred_sample = predict_pipeline(wpd_arr)
    logging.info("Potability prediction: %s", pred_sample)
    pred_dict = {"Potability": int(pred_sample)}
    return pred_dict
