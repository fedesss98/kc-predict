# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:13:47 2022

@author: Federico Amato

Read CSV data and make Pickle File
"""
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import logging

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def get_raw_data(fname):  # sourcery skip: use-contextlib-suppress
    if str(fname).endswith(".csv"):
        df = pd.read_csv(
            fname,
            sep=";",
            decimal=",",
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            dayfirst=True,
        )
    elif str(fname).endswith(".xlsx"):
        df = pd.read_excel(fname, decimal=",", index_col=0, parse_dates=True)
    try:
        df["SWC"] = df["SWC"].replace(0, np.NaN)
        df["Week"] = pd.Series(df.index).dt.isocalendar().week.values
        df["Month"] = df.index.month
        # NaN values in Precipitations and Irrigation are considered as null
        df["P"] = df["P"].fillna(0)
        df["I"] = df["I"].fillna(0)
    except KeyError:
        pass
    # Some column of the file may use dots as decimal
    # separators, so those columns are interpreted as
    # object type and must be casted to floats.
    # See for example SWC, Tavg, RHavg
    return df.astype(np.float64)


def make_pickle(df, out):
    try:
        df.to_pickle(out)
    except Exception:
        print("Something went wrong writing Pickle file.\nTry again")


def main(input_file, output_file, visualize=True):
    """
    Load raw data from input file, save it to output file, and optionally visualize it.

    :param input_file: Path to input file containing raw data
    :param output_file: Path to output file where data will be saved
    :param visualize: Whether to visualize the data (default: True)
    """
    logging.info(f'\n\n{"-"*5} MAKE DATA {"-"*5}\n\n')

    # Load raw data from input file
    data = get_raw_data(input_file)

    # Log information about the data
    logging.info(
        f"The file:\n" f"{input_file}\n" f"has the shape {data.shape} with columns:"
    )

    for c in data.columns:
        logging.info(c)

    # Save data to output file
    make_pickle(data, output_file)

    # Optionally visualize the data
    if visualize:
        data.plot(subplots=True, figsize=(10, 16))
        savepath = ROOT_DIR / "visualization/data/raw_data.png"
        if savepath.exists():
            plt.savefig(ROOT_DIR / "visualization/data" / "raw_data.png")
        else:
            plt.show()
            logging.info("Visualization folder not found. Skipping visualization.")

    logging.info(f'\n\n{"-"*21}')
    return None


if __name__ == "__main__":
    input_file = ROOT_DIR / "data/raw/db_villabate_deficit_6.csv"
    output_file = ROOT_DIR / "data/interim/data.pickle"
    visualize = True
    main(input_file, output_file, visualize=visualize)
