# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:51:27 2022

@author: Federico Amato

All the pipeline to predict Kc
"""
import pandas as pd

from data.make_data import main as make_data
from data.preprocess import main as preprocess_data
from models.make_model import ModelTrainer
from prediction.predict import main as predict
from prediction.polish import main as polish
from models.calc_metrics import main as calc_metrics

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# %% PARAMETERS

m0 = [
    "ETo",
    "U2",
    "RHmin",
    "RHmax",
    "Tmin",
    "Tmax",
    "SWC",
    "NDVI",
    "NDWI",
    "DOY",
    "I",
    "P",
    "EToC",
    "IC",
    "PC",
    "LID",
    "LPD",
    "LWD",
]
m1 = ["Rs", "U2", "RHmin", "RHmax", "Tmin", "Tmax", "SWC", "NDVI", "NDWI", "DOY"]
m2 = ["Rs", "U2", "RHmax", "Tmin", "Tmax", "SWC", "NDVI", "NDWI", "DOY"]
m3 = ["Rs", "U2", "RHmax", "Tmax", "SWC", "NDVI", "NDWI", "DOY"]
m4 = ["Rs", "U2", "RHmax", "Tmax", "SWC", "NDWI", "DOY"]
m5 = ["Rs", "U2", "Tmax", "SWC", "NDWI", "DOY"]
m6 = ["Rs", "U2", "Tmax", "SWC", "DOY"]
m7 = ["Rs", "Tmax", "SWC", "DOY"]
m8 = ["Rs", "U2", "RHmin", "RHmax", "Tmin", "Tmax"]
m9 = ["ETo", "SWC", "NDVI", "NDWI", "DOY"]
m10 = ["ETo", "NDVI", "NDWI", "DOY"]
m11 = ["Rs", "SWC", "NDVI", "NDWI", "DOY"]
m12 = ["Rs", "NDVI", "NDWI", "DOY"]
m13 = [
    "Rs",
    "U2",
    "RHmin",
    "RHmax",
    "Tmin",
    "Tmax",
    "SWC",
    "NDVI",
    "NDWI",
    "DOY",
    "I",
    "P",
]
m14 = ["Rs", "U2", "RHmin", "RHmax", "Tmin", "Tmax", "DOY"]


FEATURES = {
    "model 1": m1,
    "model 2": m2,
    "model 3": m3,
    "model 4": m4,
    "model 5": m5,
    "model 6": m6,
    "model 7": m7,
    "model 8": m8,
    "model 9": m9,
    "model 10": m10,
    "model 11": m11,
    "model 12": m12,
    "Final model": m14,
}

MAKE_DATA_PARAMETERS = {
    "input_file": "G:/UNIPA/Dropbox/crop_coefficient/data/raw/data.xlsx",
    "output_file": ROOT_DIR / "data/interim/data.pickle",
    "visualize": False,
}

PREPROCESS_PARAMETERS = {
    "input_file": ROOT_DIR / "data/interim/data.pickle",
    "scaler": "MinMax",
    "folds": 5,
    "k_seed": 24,  # 24
    "output_file": ROOT_DIR / "data/processed/processed.pickle",
    "visualize": False,
}

MODELS = {
    "rf": RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        random_state=12,
        ccp_alpha=0.0,
    ),
    'mlp': MLPRegressor(
        hidden_layer_sizes=(100, 100, 100),
        max_iter=1000,
        random_state=32652,  # 32652
    ),
    'knn': KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        ),
}

PREDICTION_PARAMETERS = {
    "output": ROOT_DIR / "data/predicted" / "predicted.pickle",
    "visualize": False,
}

POSTPROCESS_PARAMETERS = {
    "contamination": 0.01,
    "seed": 352,
    "visualize": False,
}


# %% MAIN
def main(features_set, **kwargs):
    make_data(**MAKE_DATA_PARAMETERS)

    features = FEATURES[features_set]
    PREPROCESS_PARAMETERS["features"] = features
    preprocess_data(**PREPROCESS_PARAMETERS)

    predictions = {}

    for model_name in MODELS.keys():
        model_name_to_save = f"{model_name.upper()}"
        MODEL_PARAMETERS = {
            "model": MODELS[model_name],
            "model_name": model_name_to_save,
            "features": features,
            "visualize_error": False,
        }
        trainer = ModelTrainer(**MODEL_PARAMETERS)

        PREDICTION_PARAMETERS["features"] = features
        predict(model=f"{model_name_to_save}.joblib", **PREDICTION_PARAMETERS)

        POSTPROCESS_PARAMETERS["outfile"] = f"{model_name_to_save}_kc_postprocessed"
        kc_postprocessed = polish(**POSTPROCESS_PARAMETERS)

        predictions[model_name.upper()] = kc_postprocessed

    kc_postprocessed = pd.concat(predictions, axis=1)
    calc_metrics(kc_postprocessed)


# %% Entry point
if __name__ == "__main__":
    features_set = "Final model"
    main(features_set)
