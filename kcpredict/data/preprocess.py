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
    for k, [train_index, test_index] in enumerate(folds.split(df)):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        make_pickle(train, ROOT_DIR / "data/processed" / f"train_fold_{k}.pickle")
        make_pickle(test, ROOT_DIR / "data/processed" / f"test_fold_{k}.pickle")


def scale_data(df, scaler):
    """
    Scale data using selected scaler:
    - Standard Scaler (zero mean and unit variance) [DEFAULT]
    - MinMax Scaler (values between minus one and plus one)
    """
    try:
        # Fit and save scaler
        scaler.fit(df)
        joblib.dump(scaler, ROOT_DIR / "models" / "scaler.joblib")
        scaled_values = scaler.transform(df)
        scaled_data = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
    except Exception as e:
        print(f"Error with the scaler: {str(e)}")
    return scaled_data


# def write_log(input_file, scaler, output_file, visualize, df, train, test):
#     json_log = {
#         "input_file": input_file,
#         "output_file": output_file,
#         "scaler": scaler,
#         "database_shape": df.shape(),
#         "train_set_length": len(train),
#         "test_set_length": len(test),
#     }
#     json.dump(json_log, ROOT_DIR/'docs'/'run.json')


def main(input_file, scaler, folds, k_seed, output_file, features=None, visualize=True, root_folder=ROOT_DIR):
    logging.info(f'\n\n{"-"*5} PREPROCESSING {"-"*5}\n\n')
    logging.info(f"Preprocessing file:\n{input_file}")

    if not isinstance(root_folder, Path):
        root_folder = Path(root_folder)
    input_file = root_folder / input_file
    output_file = root_folder / output_file

    # Load and preprocess data
    data = get_data(input_file)
    df = make_dataframe(data, features)
    scaler = make_scaler(scaler)

    # Visualize data
    if visualize:
        df.plot(subplots=True, figsize=(10, 16))
        plt.show()

    # Impute missing values
    df = impute_features(df, features)

    # Save and visualize imputed data
    make_pickle(df, ROOT_DIR / "data/interim" / "imputed.pickle")
    if visualize:
        df.plot(subplots=True, figsize=(10, 16))
        plt.savefig(ROOT_DIR / "visualization/data" / f"processed_{NN}_{scaler}.png")
        plt.show()

    # Save data to predict
    predict = df.loc[~df.index.isin(df.dropna().index), features]
    predict = scale_data(predict, scaler)
    make_pickle(predict, ROOT_DIR / "data/processed" / "predict.pickle")

    # Split data into train and test sets
    split_folds(df, folds, k_seed)

    # Iterate over folds
    for k in range(folds):
        train_file = ROOT_DIR / "data/processed" / f"train_fold_{k}.pickle"
        test_file = ROOT_DIR / "data/processed" / f"test_fold_{k}.pickle"

        train = pd.read_pickle(train_file)
        test = pd.read_pickle(test_file)

        # Scale fold
        train = scale_data(train, scaler)
        test = scale_data(test, scaler)

        # Save fold
        make_pickle(train, train_file)
        make_pickle(test, test_file)

    # write_log(input_file, scaler, output_file, visualize, df, train, test)

    # Scale and save final data
    df = scale_data(df, scaler)
    make_pickle(df, output_file)

    logging.info(f'\n\n{"/"*30}')


if __name__ == "__main__":
    main()
