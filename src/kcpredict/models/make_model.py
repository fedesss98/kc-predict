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
from joblib import dump, load  # To save model

# from .. neptune.log_neptune import log_neptune

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.parent


def train_model(model, k):
    train = pd.read_pickle(ROOT_DIR/'data/processed'/f'train_fold_{k}.pickle')
    X_train = train.iloc[:, :-2]
    y_train = train.iloc[:, -1].values.ravel()
    model.fit(X_train, y_train)
    return model


def test_model(model, k):
    test = pd.read_pickle(ROOT_DIR/'data/processed'/f'test_fold_{k}.pickle')
    X_test, y_test = test.iloc[:, :-2], test.iloc[:, -1]
    r2 = model.score(X_test, y_test)
    print(f'R2 score on test {k}: {r2:.4f}')
    return r2


def save_model(model_name, model):
    dump(model, ROOT_DIR/'models'/f'{model_name}.joblib')
    return None


def log_run(model, size, scores):
    print(model)
    print(f'Scores Mean: {scores.mean():.4f}')
    print(f'Score Variance: {scores.var():.4f}')
    return None


def log_model_scores(scores):
    scores = pd.DataFrame(scores, columns=['Test Scores'])
    scores.to_csv(ROOT_DIR/'logs\models_scores.csv')
    return None


def main(model, model_name, *args, **kwargs):
    print(f'\n\n{"-"*5} MODEL TRAINING {"-"*5}\n\n')
    # Find folds data
    k = len(list(ROOT_DIR.glob('data/processed/test_fold_*')))
    models = dict()
    scores = np.zeros(k)
    for fold in range(k):
        models[fold] = train_model(model, fold)
        test_score = test_model(model, fold)
        scores[fold] = test_score
    log_model_scores(scores)
    print(f'Scores Mean: {scores.mean():.4f}')
    print(f'Score Variance: {scores.var():.4f}')
    # Save the best scoring model
    best_model = models[np.argmax(scores)]
    save_model(model_name, best_model)
    # if log:
    #     log_run(model, np.array(scores), **kwargs)
    print(f'\n\n{"-"*30}\n\n')


@click.group()
@click.option('--log', is_flag=True, help='Log training to Neptune')
def make_model(*args, **kwargs):
    """
    Train a Machine Learning model and test it.
    Training and testing is implemented on k-folds of data.
    The mean score (R2) and score variance are returned.
    """
    return None


@click.command()
@click.option('--bootstrap', default=True)
@click.option('--ccp-alpha', type=click.FLOAT, default=0.0)
@click.option('--max-depth', type=click.INT, default=None)
@click.option('--max-samples', type=click.INT, default=None)
@click.option('-n', '--n-estimators', type=click.INT, default=100)
@click.option('--random-state', type=click.INT, default=6474)
def rf(**kwargs):
    model = RandomForestRegressor(**kwargs)
    model_name = 'rf'
    for key, value in model.get_params().items():
        print(f'{key:25} : {value}')
    main(model, model_name)


@click.command()
@click.option('--activation', default='relu',
              type=click.Choice(
                  ['identity', 'logistic', 'tanh', 'relu'], 
                  case_sensitive=False))
@click.option('--solver', default='adam',
              type=click.Choice(
                  ['lbfgs', 'sgd', 'adam'], 
                  case_sensitive=False))
@click.option('--alpha', type=click.FLOAT, default=0.0001)
@click.option('--learning-rate', default='constant',
              type=click.Choice(
                  ['constant', 'invscaling', 'adaptive'], 
                  case_sensitive=False))
@click.option('--max-iter', type=click.INT, default=200)
@click.option('--shuffle', default=True)
@click.option('-hls', '--hidden-layer-sizes', 
              type=click.INT, default=[100,], multiple=True)
@click.option('--random-state', type=click.INT, default=12)
def mlp(**kwargs):
    model = MLPRegressor(**kwargs)
    model_name = 'mlp'
    for key, value in model.get_params().items():
        print(f'{key:25} : {value}')
    main(model, model_name)
    return None


@click.command()
@click.option('--kernel', default='rbf',
              type=click.Choice(
                  ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 
                  case_sensitive=False))
@click.option('--degree', type=click.INT, default=3)
@click.option('--gamma', default='scale',
              type=click.Choice(
                  ['scale', 'auto',], 
                  case_sensitive=False))
@click.option('--tol', type=click.FLOAT, default=1e-3)
@click.option('--epsilon', type=click.FLOAT, default=0.1)
@click.option('--max-iter', type=click.INT, default=-1)
def svr(**kwargs):
    model = SVR(**kwargs)
    model_name = 'mlp'
    for key, value in model.get_params().items():
        print(f'{key:25} : {value}')
    main(model, model_name)  
    return None


@click.command()
@click.option('--kernel', default=None,)
@click.option('--alpha', type=click.FLOAT, default=1e-10)
@click.option('--random-state', type=click.INT, default=6474)
def gpr(**kwargs):
    model = GaussianProcessRegressor(**kwargs)
    model_name = 'mlp'
    for key, value in model.get_params().items():
        print(f'{key:25} : {value}')
    main(model, model_name)  
    return None


make_model.add_command(rf)
make_model.add_command(mlp)
make_model.add_command(svr)
make_model.add_command(gpr)

if __name__ == "__main__":
    rf()
