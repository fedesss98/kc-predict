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
import shutil

from data.make_data import main as make_data
from data.preprocess import main as preprocess_data
from models.make_model import ModelTrainer
from prediction.predict import main as predict
from prediction.polish import main as polish
from prediction.make_trapezoidal import main as make_trapezoidal
from models.calc_metrics import main as calc_metrics

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


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
        self.project_dir = self.setup_project()

        # Add the root folder to the configuration
        for key in ["make-data", "preprocess", "prediction", "postprocess"]:
            try:
                self.config[key]["root_folder"] = self.project_dir
            except KeyError:
                logging.warning(f"Key {key} not found in configuration file")

        # Read and Preprocess data
        make_data(**self.config["make-data"])
        preprocess_data(**self.config["preprocess"])

        # Save features to be used
        self.features = self.config["preprocess"]["features"]

        # Initialize models
        self._models = {}
        self.models = self.config["models"]

        # Train and save the best models on a KFold cross-validation
        self.make_model()
        #Predict ETa using the best models and make the Kc series
        self.predict_eta()

        # Postprocess the predictions
        self.kc_postprocessed = polish(**self.config["postprocess"])
        self.kc_trapezoidal = make_trapezoidal(**self.config["postprocess"])

        print("END")

    def setup_project(self):
        """
        Set the project directory where all the data will be saved.
        If the directory does not exist, it will be created.
        If the directory is not specified in the configuration file, it will be set to the root directory.
        The source dataset is copy-pasted from the root directory, if the dataset is not already in the project directory.
        """
        project_dir = Path(self.config.get("project_dir", self.root))
        if not os.path.isdir(project_dir):
            os.makedirs(project_dir)
            logging.info(f"Created project folder: {project_dir}")
        else:
            logging.info(f"Project folder set to: {project_dir}")

        raw_data_file = Path(self.config["make-data"]["input_file"])

        # Check if all needed directories exist, otherwise create them
        for key in ["make-data", "preprocess", "prediction", "postprocess"]:
            if self.config[key].get("output_path", None):
                output_dir = project_dir / Path(self.config[key]["output_path"])
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                    logging.info(f"Created output folder: {output_dir}")

        # Check if the raw data file is not already in the project directory
        if not os.path.isfile(project_dir / raw_data_file):
            if not os.path.isfile(self.root / raw_data_file):
                raise FileNotFoundError(
                    "Raw data file not found in project directory or root directory"
                )

            # First check if destination directory exists
            if not os.path.isdir(project_dir / raw_data_file.parent):
                os.makedirs(project_dir / raw_data_file.parent)
                logging.info(f"Created data folder: {project_dir / raw_data_file.parent}")
            # Copy the raw data file to the project directory
            shutil.copy(self.root / raw_data_file, project_dir / raw_data_file)
            logging.info(f"Copied raw data file to project directory: {project_dir}")
        # Copy the configuration file to the project directory
        try:
            shutil.copy(self.root / "config.toml", project_dir / "config.toml")
        except shutil.SameFileError:
            logging.warning("Configuration file already in project directory")

        # Create a visualization folder if it does not exist
        visualization_dir = project_dir / "visualization"
        if not os.path.isdir(visualization_dir):
            os.makedirs(visualization_dir)
            logging.info(f"Created visualization folder: {visualization_dir}")

        # Create a models folder if it does not exist
        models_dir = project_dir / "models"
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir)
            logging.info(f"Created models folder: {models_dir}")

        # Create an external data folder if the trapezoidal file is given
        if self.config["postprocess"].get("trapezoidal_path", None):
            external_data_dir = project_dir / "data" / "external"
            if not os.path.isdir(external_data_dir):
                os.makedirs(external_data_dir)
                logging.info(f"Created external data folder: {external_data_dir}")
            # Copy the trapezoidal file to the external data folder
            trapezoidal_file = Path(self.config["postprocess"]["trapezoidal_path"])
            if not trapezoidal_file.is_file():
                trapezoidal_file = '../' / trapezoidal_file
            shutil.copy(trapezoidal_file, external_data_dir / trapezoidal_file.name)
            logging.info(f"Copied trapezoidal file to external data folder: {external_data_dir}")

        return project_dir

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models_config) -> dict:
        models_map = {
            "rf": RandomForestRegressor,
            "mlp": MLPRegressor,
            "knn": KNeighborsRegressor,
        }
        for model_name, model_config in models_config.items():
            self._models[model_name] = models_map[model_name](**model_config)
        return self._models

    def make_model(self):
        for model_name, model in self.models.items():
            model_kwargs = dict(
                model=model, model_name=model_name, features=self.features,
                **self.config["prediction"]
            )
            # Train the model on each fold and save the best-performing one
            trainer = ModelTrainer(**model_kwargs)
            trainer.train_on_folds()

    def predict_eta(self):
        # predict(model=f"{model_name_to_save}.joblib", **PREDICTION_PARAMETERS)
        for model_name in self.models.keys():
            predict(model_name, features=self.features, **self.config["prediction"])


def main(root):
    predictor = KcPredictor(root)


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
