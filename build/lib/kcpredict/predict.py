# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 18:43:00 2022

@author: Federico Amato

Predict ETa with saved model.
From measured and predicted ETa computes KC dividing by measured ET0.

"""
import click
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load  # To save model

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent


def ask_for_model():
    saved_models = list(ROOT_DIR.glob('models/*'))
    print(f"You have {len(saved_models)} saved:")
    for m in saved_models:
        print(m)
    model_name = input("Which model do you want to use? ")
    return str(model_name)


def load_model(model_name):
    if model_name is None:        
        model_name = ask_for_model()
    model = load(ROOT_DIR/'models'/f'{model_name}')
    return model


def main(model, visualize):
    # Features to predict ETa
    df = pd.read_pickle(ROOT_DIR/'data/processed'/'predict.pickle')
    X = df.iloc[:, :-1]
    eto = df['ET0']
    try:
        model = load_model(model)
        eta = model.predict(X)
    except FileNotFoundError:
        print("Error finding the model. Remember to include file extension.")
    if visualize:
        eta.plot()
        plt.show()


@click.command()
@click.option('-m', '--model', default=None, help="Name of saved model")
@click.option('-v', '--visualize', is_flag=True)
def predict(model, visualize):
    """
    Predict ETa with given model.
    From measured and predicted ETa computes KC dividing by measured ET0.
    """
    main(model, visualize)

if __name__ == "__main__":
    predict()



