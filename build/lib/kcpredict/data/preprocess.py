# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:03:31 2022

@author: Federico Amato

Preprocess ETa data for predictions.

- Select features
- Impute missing values with KNNImputer
- Scale data with StandardScaler or MinMaxScaler

"""
import click
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from make_data import get_features, get_target, make_pickle

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent

NN = 5  # Number of neighboring samples to use for imputation
    

def get_data(fname):
    df = pd.read_pickle(fname)
    return df


def make_dataframe(data):
    """Selects only relevant features (Model #1 in previous experiments)"""
    features = get_features(data)
    target = get_target(data)
    # Concatenate features and target data
    df = pd.concat([features, target], axis=1)
    return df

def impute_features(df):
    features = get_features(df)
    # Impute missing features
    imputer = KNNImputer(n_neighbors=NN, weights='distance')
    imputed_values = imputer.fit_transform(features)
    # Recreate imputed DataFrame inserting target column
    # Take numpy array of ETa values
    eta_values = df['ETa'].values.reshape(-1, 1)
    # and merge it with imputed feature values
    imputed_values = np.append(imputed_values, eta_values, axis=1)
    # Make DataFrame
    imputed_data = pd.DataFrame(imputed_values, 
                                columns=df.columns, index=df.index)
    return imputed_data

def scale_data(df, scaler):
    """
    Scale data using selected scaler:
    - Standard Scaler (zero mean and unit variance) [DEFAULT]
    - MinMax Scaler (values between minus one and plus one)
    """
    if scaler == 'Standard':
        scaler = StandardScaler()
    elif scaler == 'MinMax':
        scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_values = scaler.fit_transform(df)
    scaled_data = pd.DataFrame(scaled_values, 
                               columns=df.columns, index=df.index)
    return scaled_data

@click.command()
@click.option('-in', '--input-file',
              type=click.Path(),
              default=(ROOT_DIR /'data/interim'/'db_villabate.pickle'),
              )
@click.option('-s', '--scaler', default='MinMax', 
              type=click.Choice(['Standard', 'MinMax'], case_sensitive=False))
@click.option('-out', '--output-file', 
              type=click.Path(),
              default=(ROOT_DIR/'data/processed'/'processed.pickle'),
              )
# Optionally save the plot of the dataframe
@click.option('-v', '--visualize', is_flag=True,)
def main(input_file, scaler, output_file, visualize):
    print(f'\n\n{"-"*5} PREPROCESSING {"-"*5}')
    print(input_file)
    data = get_data(input_file)
    df = make_dataframe(data)
    df.plot(subplots=True, figsize=(10, 16))
    if visualize:
        plt.show()        
    # IMPUTE
    df = impute_features(df)
    # SCALE
    df = scale_data(df, scaler)
    # SAVE AND VISUALIZE
    make_pickle(df, output_file) 
    df.plot(subplots=True, figsize=(10, 16))
    if visualize:
        plt.savefig(ROOT_DIR/
                    'visualization/data'/
                    f'processed_{NN}_{scaler}.png')
        plt.show()
    print(f'\n\n{"-"*11}')
    

if __name__ == "__main__":
    main()