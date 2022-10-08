# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 22:00:00 2022

@author: Federico Amato

Train a model and test it.
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load  # To save model

from .. neptune.log_neptune import log_neptune

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.parent



def get_features(df):
    features = [
        'Rs', 'U2', 'RHmin', 'RHmax', 'Tmin', 
        'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'
        ]
    return df.loc[:, features]


def get_target(df):
    target = ['ETa']
    return df.loc[:, target]


def get_data():
    df = pd.read_pickle(ROOT_DIR/'data/processed'/'processed.pickle')
    return df


def make_sets(df):
    df = df.dropna()
    X = get_features(df)
    y = get_target(df)
    train_test_sets = train_test_split(X, y, test_size=0.2, random_state=1)
    return train_test_sets


def setup_model(model, size):
    model = "".join(model)
    if model == 'rf':
        scikit_model = RandomForestRegressor(n_estimators=size)
    elif model == 'mlp':
        scikit_model = MLPRegressor(hidden_layer_sizes=size)
    else:
        raise Exception('Model not found.')
    for key, value in scikit_model.get_params().items():
        print(f'{key:25} : {value}')
    return scikit_model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train.values.ravel())
    return model


def test_model(model, X_test, y_test):
    r2 = model.score(X_test, y_test)
    print(f'\nR2 score on test: {r2:.4f}')
    return r2


def save_model(model):
    dump(model, ROOT_DIR/'models'/'rf.joblib')


def log_run(run):
    print(run)


def main(model, size, log):
    run = {"tags": ["ml", "crop_coefficient"]}
    print(f'\n\n{"-"*5} {model.upper()} MODEL TRAINING {"-"*5}\n\n')
    df = get_data()
    X_train, X_test, y_train, y_test = make_sets(df)
    
    model = setup_model(model, size)

    model = train_model(model, X_train, y_train)
    run["test_score"] = test_model(model, X_test, y_test)  # type: ignore
    save_model(model)
    if log:
        log_run(run)
    
    print(f'\n\n{"-"*30}\n\n')

@click.command()
# Use Random Forest
@click.option('-rf', '--random-forest', 'model', flag_value='rf', default=True)
# Use Neural Network
@click.option('-mlp', '--multi-layer-perceptron', 'model', flag_value='mlp')
# Size of the forest/network
@click.option('--size', default=100, 
              help='Size of the forest/network')
@click.option('--log', is_flag=True,
              help='Log training to Neptune')
def make_model(model, size, log):
    """
    Train a Machine Learning model and test it.
    """
    main(model, size, log)
    return None


if __name__ == "__main__":
    make_model()
