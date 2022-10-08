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
ROOT_DIR = Path(__file__).parent.parent.parent


def get_features(df):
    features = [
        'Rs', 'U2', 'RHmin', 'RHmax', 'Tmin', 
        'Tmax', 'SWC', 'NDVI', 'NDWI', 'DOY'
        ]
    return df.loc[:, features]

def get_target(df):
    target = ['ETa']
    return df.loc[:, target]


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
    

# Click module enable command line optional arguments
@click.command()
# Input file name can be given in command line
@click.option('-in', '--input-file',
              type=click.Path(),
              default=(ROOT_DIR/'data/interim'/'db_villabate_deficit_6.csv'),
             )
# Outuput file name can be given in command line
@click.option('-out', '--output-file', 
              type=click.Path(),
              default=(ROOT_DIR/'data/interim'/'db_villabate.pickle'),
             )
# Optionally save the plot of the dataframe
@click.option('-v', '--visualize', is_flag=True,)
def main(input_file, output_file, visualize):
    print(f'\n\n{"-"*5} MAKE DATA {"-"*5}\n\n')
    data = get_csv_data(input_file)
    print(data)
    make_pickle(data, output_file)
    if visualize:
        data.plot(subplots=True, figsize=(10, 16))
        plt.savefig(ROOT_DIR/'visualization/data'/'raw_data.png')
    print(f'\n\n{"-"*21}')
    

if __name__ == "__main__":
    main()