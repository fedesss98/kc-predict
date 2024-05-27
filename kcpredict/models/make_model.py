# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 22:00:00 2022

@author: Federico Amato

Train a model and test it.
Validation on 2022 measures.
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load  # To save model

import logging

# from .. neptune.log_neptune import log_neptune

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_squared_error, r2_score

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


class ModelTrainer:
    def __init__(self, model, model_name, features, input, output, root_folder=ROOT_DIR, **kwargs):
        logging.info(f'\n\n{"-"*7} {model_name.upper()} MODEL TRAINING {"-"*7}\n\n')

        self.root_folder = Path(root_folder)
        self.input_folder = root_folder / input
        self.output_folder = root_folder / output

        self.features = features
        self.model = model
        self.model_name = model_name


        # Find number of folds
        self.k = len(list(self.input_folder.glob("test_fold_*")))
        if self.k == 0:
            raise FileNotFoundError(f"No folds data found in input folder {self.input_folder}")
        self.scores = np.zeros((self.k, 2))
        # For each fold one model is trained
        self.trained_models = {}

    def train_on_folds(self):
        for fold in range(self.k):
            trained_model = self.train_model(self.model, fold)
            self.trained_models[fold] = trained_model
            self.scores[fold] = self.test_model(trained_model, fold)

        # Save scores to file
        np.savetxt(self.output_folder / f"{self.model_name}_scores.csv", self.scores, delimiter=";")

        logging.info(f"R2 Scores Mean: {self.scores.mean(axis=0)[0]:.2f}")
        logging.info(f"R2 Scores Max: {self.scores.max(axis=0)[0]:.2f}")
        logging.info(f"Score Variance: {self.scores.var(axis=0)[0]:.4f}")

        # Save the best scoring model
        self.best_model = self.trained_models[np.argmax(self.scores, axis=0)[0]]
        self.save_model()

        # Print ending string
        logging.info(f'\n\n{"/"*30}\n')

    def train_model(self, model, k):
        train = pd.read_pickle(self.input_folder / f"train_fold_{k}.pickle")
        X_train = train.loc[:, self.features]
        y_train = train.loc[:, "ETa"].values.ravel()
        model.fit(X_train, y_train)
        return model

    def test_model(self, model, k):
        test = pd.read_pickle(self.input_folder / f"test_fold_{k}.pickle")
        X_test, y_test = test.loc[:, self.features], test.loc[:, "ETa"]
        prediction_test = model.predict(X_test)
        # Compute scores on scaled values
        r2_scaled = r2_score(y_test, prediction_test)
        rmse_scaled = mean_squared_error(y_test, prediction_test, squared=False)
        # Rescale values
        prediction_test = self.rescale_eta(prediction_test)
        measures_test = self.rescale_eta(y_test.values, y_test.index).squeeze()
        # Compute scores on rescaled values
        r2 = r2_score(measures_test, prediction_test)
        rmse = mean_squared_error(measures_test, prediction_test, squared=False)

        logging.info(
            f"R2 score on test {k}: {r2_scaled:.2f} - {r2:.2f}"
            f"\nRMSE score on test: {rmse_scaled:.2f} - {rmse:.2f}"
        )
        return r2, rmse

    def save_model(self):
        dump(self.best_model, self.root_folder / f"models/{self.model_name}.joblib")
        return None

    def rescale_eta(self, eta, index=None):
        # Create fake DataFrame with fake features
        X = pd.DataFrame(columns=self.features)
        X["ETa"] = eta
        scaler = load(self.root_folder / "models/scaler.joblib")
        rescaled_eta = scaler.inverse_transform(X)[:, [-1]].ravel()
        if index is not None:
            # Create a DataFrame
            rescaled_eta = pd.DataFrame(rescaled_eta, columns=["ETa"], index=index)
        return rescaled_eta


@click.group()
@click.option("--log", is_flag=True, help="Log training to Neptune")
def make_model(*args, **kwargs):
    """
    Train a Machine Learning model and test it.
    Training and testing is implemented on k-folds of data.
    The mean score (R2) and score variance are returned.
    """
    return None


@click.command()
@click.option("--bootstrap", default=True)
@click.option("--ccp-alpha", type=click.FLOAT, default=0.0)
@click.option("--max-depth", type=click.INT, default=None)
@click.option("--max-samples", type=click.INT, default=None)
@click.option("-n", "--n-estimators", type=click.INT, default=100)
@click.option("--random-state", type=click.INT, default=6474)
def rf(**kwargs):
    model = RandomForestRegressor(**kwargs)
    model_name = "rf"
    for key, value in model.get_params().items():
        print(f"{key:25} : {value}")
    model = ModelTrainer(model, model_name)


@click.command()
@click.option(
    "--activation",
    default="relu",
    type=click.Choice(["identity", "logistic", "tanh", "relu"], case_sensitive=False),
)
@click.option(
    "--solver",
    default="adam",
    type=click.Choice(["lbfgs", "sgd", "adam"], case_sensitive=False),
)
@click.option("--alpha", type=click.FLOAT, default=0.0001)
@click.option(
    "--learning-rate",
    default="constant",
    type=click.Choice(["constant", "invscaling", "adaptive"], case_sensitive=False),
)
@click.option("--max-iter", type=click.INT, default=200)
@click.option("--shuffle", default=True)
@click.option(
    "-hls",
    "--hidden-layer-sizes",
    type=click.INT,
    default=[
        100,
    ],
    multiple=True,
)
@click.option("--random-state", type=click.INT, default=12)
def mlp(**kwargs):
    model = MLPRegressor(**kwargs)
    model_name = "mlp"
    for key, value in model.get_params().items():
        print(f"{key:25} : {value}")
    model = ModelTrainer(model, model_name)
    return None


@click.command()
@click.option(
    "--kernel",
    default="rbf",
    type=click.Choice(
        ["linear", "poly", "rbf", "sigmoid", "precomputed"], case_sensitive=False
    ),
)
@click.option("--degree", type=click.INT, default=3)
@click.option(
    "--gamma",
    default="scale",
    type=click.Choice(
        [
            "scale",
            "auto",
        ],
        case_sensitive=False,
    ),
)
@click.option("--tol", type=click.FLOAT, default=1e-3)
@click.option("--epsilon", type=click.FLOAT, default=0.1)
@click.option("--max-iter", type=click.INT, default=-1)
def svr(**kwargs):
    model = SVR(**kwargs)
    model_name = "mlp"
    for key, value in model.get_params().items():
        print(f"{key:25} : {value}")
    model = ModelTrainer(model, model_name)
    return None


@click.command()
@click.option(
    "--kernel",
    default=None,
)
@click.option("--alpha", type=click.FLOAT, default=1e-10)
@click.option("--random-state", type=click.INT, default=6474)
def gpr(**kwargs):
    model = GaussianProcessRegressor(**kwargs)
    model_name = "mlp"
    for key, value in model.get_params().items():
        print(f"{key:25} : {value}")
    model = ModelTrainer(model, model_name)
    return None


make_model.add_command(rf)
make_model.add_command(mlp)
make_model.add_command(svr)
make_model.add_command(gpr)

if __name__ == "__main__":
    rf()
