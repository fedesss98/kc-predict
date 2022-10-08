# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 22:00:00 2022

@author: Federico Amato

Train a model and test it.
"""
import click
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load  # To save model

# from .. neptune.log_neptune import log_neptune

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


def train_model(model, k):
    train = pd.read_pickle(ROOT_DIR/'data/processed'/f'train_fold_{k}.pickle')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1].values.ravel()
    model.fit(X_train, y_train)
    return model


def test_model(model, k):
    test = pd.read_pickle(ROOT_DIR/'data/processed'/f'test_fold_{k}.pickle')
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    r2 = model.score(X_test, y_test)
    print(f'R2 score on test {k}: {r2:.4f}')
    return r2


def save_model(model):
    dump(model, ROOT_DIR/'models'/'rf.joblib')
    return None


def log_run(model, size, scores):
    print(model)
    print(f'Scores Mean: {scores.mean():.4f}')
    print(f'Score Variance: {scores.var():.4f}')
    return None


def main(model, size, log):
    print(f'\n\n{"-"*5} {model.upper()} MODEL TRAINING {"-"*5}\n\n')
    model = setup_model(model, size)
    # Find folds data
    k = len(list(ROOT_DIR.glob('data/processed/test_fold_*')))
    scores = list()
    for fold in range(k):
        model = train_model(model, fold)
        test_score = test_model(model, fold)
        scores.append(test_score)
    save_model(model)
    if log:
        log_run(model, size, np.array(scores))
    
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
    Training and testing is implemented on k-folds of data.
    The mean score (R2) and score variance are returned.
    """
    main(model, size, log)
    return None


if __name__ == "__main__":
    make_model()
