# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:13:47 2022

@author: Federico Amato

Read CSV data and make Pickle File
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.parent


def get_csv_data(fname):
    df = pd.read_csv(fname, sep=';', decimal=',',
                     index_col=0, parse_dates=True)
    # Some column of the csv use dots as decimal separators
    # so those columns are interpreted as object type and
    # must be casted to floats
    return df.astype(np.float64)


def make_pickle(df, out):
    try:
        df.to_pickle(out)
    except Exception:
        print("Something went wrong writing Pickle file.\nTry again")
    

def main(input_file, output_file, visualize):
    print(f'\n\n{"-"*5} MAKE DATA {"-"*5}\n\n')
    data = get_csv_data(input_file)
    print(f'{input_file} has the shape {data.shape} with columns:')
    for c in data.columns:
        print(c)
    make_pickle(data, output_file)
    if visualize:
        data.plot(subplots=True, figsize=(10, 16))
        plt.savefig(ROOT_DIR/'visualization/data'/'raw_data.png')
    print(f'\n\n{"-"*21}')
    return None


@click.command()
@click.option('-in', '--input-file',
              type=click.Path(),
              default=(ROOT_DIR/'data/raw'/'data.csv'),)
@click.option('-out', '--output-file', 
              type=click.Path(),
              default=(ROOT_DIR/'data/interim'/'data.pickle'),)
@click.option('-v', '--visualize', is_flag=True,)
def make_data(input_file, output_file, visualize):
    """
    Read raw CSV file and save the dataframe in a Pickle file.
    """
    main(input_file, output_file, visualize)
    return None


if __name__ == "__main__":
    make_data()