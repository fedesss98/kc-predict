# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:51:27 2022

@author: Federico Amato

All the pipeline to predict Kc
"""
import pandas as pd
import logging
import tomli
import os

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


FEATURES = {"2024_variouscrop":
                [
                    "DOY", "Tmin", "Tmax", "Tdew", "Uwind", "Vwind", "Rs",
                    # "ETo"
                ]
            }


MAKE_DATA_PARAMETERS = {
    "input_file": "G:/UNIPA/DOTTORATO/MACHINE_LEARNING/crop_coefficient/kc-predict/data/raw/data_us_arm.csv",
    "output_file": ROOT_DIR / "data/interim/data.pickle",
    "visualize": True,
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
        random_state=42,
        ccp_alpha=0.0,
    ),
    "mlp": MLPRegressor(
        hidden_layer_sizes=(100, 100, 100),
        max_iter=1000,
        random_state=32652,  # 32652
    ),
    "knn": KNeighborsRegressor(
        n_neighbors=7,
        weights="distance",
    ),
}

PREDICTION_PARAMETERS = {
    "output": ROOT_DIR / "data/predicted" / "predicted.pickle",
    "visualize": True,
}

POSTPROCESS_PARAMETERS = {
    "contamination": 0.1,
    "seed": 42,
    "visualize": True,
}


class KcPredictor:
    def __init__(self, root, *args, **kwargs):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"{root} is not a valid directory")
        self.root = Path(root)
        logging.info(f"Root directory: {self.root}")

        # Read and save TOML configuration file
        with open(self.root / "config.toml", "rb") as f:
            self.config = tomli.load(f)

        # Save or create project directory
        self.project_dire = self.setup_project()

        # Add the root folder to the configuration
        for key in ["make-data", "preprocess", "prediction"]:
            self.config[key]["root_folder"] = self.root

        # Read and Preprocess data
        make_data(**self.config["make-data"])
        preprocess_data(**self.config["preprocess"])
        
        # Save features to be used
        self.features = self.config["preprocess"]["features"]

        # Initialize models
        self._models = self.models(self.config["models"])

        print("END")


    def setup_project(self):
        project_dir = self.config.get("project-dir", self.root)
        if not os.path.isdir(project_dir):
            os.makedirs(project_dir)
            logging.info(f"Created project folder: {project_dir}")
        else:
            logging.info(f"Project folder set to: {project_dir}")
        return project_dir


    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models_config):
        models_map = {
            "rf": RandomForestRegressor,
            "mlp": MLPRegressor,
            "knn": KNeighborsRegressor,
        }
        for model_name, model_config in models_config.items():
            model = models_map[model_name]
            self._models[model_name] = model(**model_config)

        return self._models

    def predict(self):
        for model_name, model in self.models.items():
            model_kwargs = dict(
                model=model, model_name=model_name, features=self.features, root_folder=self.root
            )
            # Train the model on each fold and save the best-performing one
            # trainer = ModelTrainer(**model_kwargs)
            # trainer.train_on_folds()





def main(root):
    predictor = KcPredictor(root)


def second_main(features_set, **kwargs):
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
    # calc_metrics(kc_postprocessed)


if __name__ == "__main__":
    # Config logging module
    logging.basicConfig(
        encoding="utf-8",
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            # logging.FileHandler(ROOT_DIR / "logs/et_predict.log"),
            logging.StreamHandler(),
        ],
    )
    # root = r"G:\UNIPA\DOTTORATO\MACHINE_LEARNING\crop_coefficient\kc-predict\data\usarm_fede"
    root = ROOT_DIR
    main(root)
