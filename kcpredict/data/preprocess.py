# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:03:31 2022

@author: Federico Amato

Preprocess ETa data for predictions.

- Select features
- Impute missing values with KNNImputer
- Split data with KFolds
- Scale data with StandardScaler or MinMaxScaler

Train set must never see test set, even during the scaling.
Thus the k-folds slpit must come before the scaling.

FREE PARAMS:
    - NN: number of neighbors for KNNInputer
"""
import matplotlib.pyplot as plt
import numpy as np
import joblib  # To save scaler
import json  # To save log
import pandas as pd
import logging

from sklearn.impute import KNNImputer

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa

# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from .make_data import make_pickle

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

NN = 5  # Number of neighboring samples to use for imputation
DEFAULT_FEATURES = [
    "Rs",
    "U2",
    "RHmin",
    "RHmax",
    "Tmin",
    "Tmax",
    "SWC",
    "DOY",
    "Month",
    "Week",
    "ETo",
]


def get_data(fname):
    return pd.read_pickle(fname)


def get_features(df, features):
    if not isinstance(features, list):
        # Take default features
        features = DEFAULT_FEATURES

    # Print missing features asked in the dataframe
    for f in features:
        if f not in df.columns:
            # print(f"Feature {f} not present in raw data features!")
            raise KeyError(f"Feature {f} not present in raw data features!")

    # And select only the ones present in the dataframe
    features = [f for f in features if f in df.columns]

    return df.loc[:, features]


def get_target(df):
    target = ["ETa"]
    return df.loc[:, target]


def make_dataframe(data, features):
    """Selects only relevant features (Model #1 in previous experiments)"""
    features = get_features(data, features)
    target = get_target(data)
    return pd.concat([features, target], axis=1)


def make_scaler(scaler):
    if isinstance(scaler, str):
        if scaler == "Standard":
            scaler = StandardScaler()
        elif scaler == "MinMax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        try:
            scaler.set_params()
        except Exception as e:
            raise Exception(f"Error with the scaler:\n{str(e)}") from e
    return scaler


def impute_features(df, features):
    features = get_features(df, features)
    # Impute missing features
    # imputer = KNNImputer(n_neighbors=NN, weights='distance')
    imputer = IterativeImputer(random_state=0)
    imputed_values = imputer.fit_transform(features)
    # Recreate imputed DataFrame inserting target column
    # Take numpy array of ETa values
    eta_values = df["ETa"].values.reshape(-1, 1)
    # and merge it with imputed feature values
    imputed_values = np.append(imputed_values, eta_values, axis=1)
    return pd.DataFrame(imputed_values, columns=df.columns, index=df.index)


def split_folds(df, k, k_seed=2):
    folds = KFold(k, shuffle=True, random_state=k_seed)
    df = df.dropna()
    folds_data = {}
    for k, [train_index, test_index] in enumerate(folds.split(df)):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        folds_data[k] = (train, test)
        # make_pickle(train, output_folder / f"train_fold_{k}.pickle")
        # make_pickle(test, output_folder / f"test_fold_{k}.pickle")
    return folds_data


def scale_data(df, scaler, scaler_file=ROOT_DIR / "models" / "scaler.joblib"):
    """
    Scale data using selected scaler:
    - Standard Scaler (zero mean and unit variance) [DEFAULT]
    - MinMax Scaler (values between minus one and plus one)
    """
    try:
        # Fit and save scaler
        scaler.fit(df)
        joblib.dump(scaler, scaler_file)
        scaled_values = scaler.transform(df)
        scaled_data = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
    except Exception as e:
        print(f"Error with the scaler: {str(e)}")
    return scaled_data


def main(input, output, scaler, folds, k_seed, features=None, visualize=True, root_folder=ROOT_DIR):
    logging.info(f'\n\n{"-"*5} PREPROCESSING {"-"*5}\n\n')

    if not isinstance(root_folder, Path):
        root_folder = Path(root_folder)
    input_file = root_folder / input / "data.pickle"
    output_folder = root_folder / output


    # Load and preprocess data
    data = get_data(input_file)
    df = make_dataframe(data, features)

    scaler_name = scaler
    scaler = make_scaler(scaler_name)

    # Visualize data
    if visualize:
        df.plot(subplots=True, figsize=(10, 16), title="Raw data")
        plt.show()

    # Impute missing values
    df = impute_features(df, features)

    # Save and visualize imputed data
    make_pickle(df, output_folder / "imputed.pickle")

    # Save data to predict
    predict = df.loc[~df.index.isin(df.dropna().index), features]
    predict = scale_data(predict, scaler, root_folder / "models/scaler_predict.joblib")
    make_pickle(predict, output_folder / "predict.pickle")

    # Split data into train and test sets
    folds_data = split_folds(df, folds, k_seed)

    # Iterate over folds
    for k in range(folds):
        train_file = output_folder / f"train_fold_{k}.pickle"
        test_file = output_folder / f"test_fold_{k}.pickle"

        train, test = folds_data[k]

        # Scale fold
        train = scale_data(train, scaler, root_folder / "models/scaler_train.joblib")
        test = scale_data(test, scaler, root_folder / "models/scaler_test.joblib")

        # Save fold
        make_pickle(train, train_file)
        make_pickle(test, test_file)

    # Scale and save final data
    df = scale_data(df, scaler, root_folder / "models/scaler.joblib")
    make_pickle(df, output_folder / "preprocessed.pickle")

    # Visualize processed data
    if visualize:
        df.plot(subplots=True, figsize=(10, 16), title="Imputed and Scaled data")
        plt.savefig(root_folder / "visualization/" / f"processed_{scaler_name}.png")
        plt.show()

    logging.info(f'\n\n{"/"*30}')


if __name__ == "__main__":
    main()
