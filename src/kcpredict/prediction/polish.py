# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:00:35 2022

@author: Federico Amato

- Outliers removal
- Seasonal decomposition (noise removal)
- SWC filter
"""
import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .predict import plot_prediction

from sklearn.ensemble import IsolationForest

from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn as sns

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.parent


def remove_outliers(df, detector):
    measured = df.loc[df['Source']=='Measured']
    detector.fit(measured['Kc'].values.reshape(-1, 1))
    inliers = detector.predict(df['Kc'].values.reshape(-1, 1))
    inliers = df.loc[inliers == 1]
    print(f'Removed {len(df)-len(inliers)} outliers')
    return inliers


def remove_noise(df):
    decomposition = seasonal_decompose(df['Kc'], 
                                       model='additive', period=365)
    trend = decomposition.trend.bfill().ffill()
    mean_trend = decomposition.trend.mean()
    df_denoised = (decomposition.seasonal + mean_trend).to_frame(name='Kc')
    df_denoised['Source'] = df['Source']
    return df_denoised


def swc_filter(df):
    swc = pd.read_pickle(ROOT_DIR/'data/interim'/'data.pickle')['SWC']
    df_filtered = df.loc[swc>0.21]
    return df_filtered


def rolling_analysis(df):
    df_rolling = df['Kc'].rolling(window=30).mean().to_frame()
    df_rolling['Source'] = df['Source']
    return df_rolling


def get_trapezoidal():
    df = pd.read_csv(ROOT_DIR/'data/external'/'trapezoidal_kc.csv',
                     sep=';', decimal=',',
                     index_col=0,
                     parse_dates=True, 
                     infer_datetime_format=True, dayfirst=True,
                     skiprows=[0])
    return df


def add_plot_trapezoidal(ax):
    trapezoidal = get_trapezoidal().loc['2018-01':]
    ax.plot(trapezoidal.iloc[:, 0], ls='--', c='blue', alpha=0.8,
            label=trapezoidal.columns[0])
    ax.plot(trapezoidal.iloc[:, 1], ls=':', c='blue', alpha=0.8,
            label=trapezoidal.columns[1])
    return ax


def add_plot_measures(df, ax):
    x = df.loc[df['Source']=='Measured', 'Kc'].index
    y = df.loc[df['Source']=='Measured', 'Kc'].values
    ax.scatter(x, y, s=5,
               label='Measured', c='r')
    return ax

def add_plot_predictions(df, ax):
    x = df.loc[df['Source']=='Predicted', 'Kc'].index
    y = df.loc[df['Source']=='Predicted', 'Kc'].values
    ax.scatter(x, y, s=5,
               label='Predicted', c='k')
    return ax


def add_plot_sma(df, ax):
    x = df.index
    y = df['Kc'].rolling(30).mean().values
    ax.plot(x, y, c='g',
            label='Simple Moving Average',)
    return ax
    

# %%
def make_plot(df, trapezoidal=True, measures=True, sma=True, predictions=True):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title('Kc Predictions')
    
    if trapezoidal:
        ax = add_plot_trapezoidal(ax)
    if measures:
        ax = add_plot_measures(df, ax)
    if predictions:
        ax = add_plot_predictions(df, ax)
    if sma:
        ax = add_plot_sma(df, ax)
    
    ax.set_ylim(0, 1.4)
    ax.legend()
    ax.grid(axis='x', linestyle='--')
    plt.show()
    return ax

# %%
def main(visualize):
    kc = pd.read_pickle(ROOT_DIR/'data/predicted'/'predicted.pickle')
    plot_prediction(kc, 'Kc')
    
    detector = IsolationForest(contamination=0.01, random_state=352)
    kc_inlier = remove_outliers(kc, detector)
    kc_denoised = remove_noise(kc_inlier)    
    kc_filtered = swc_filter(kc_denoised)
    
    if visualize:
        plot_prediction(kc_inlier, 'Kc',  title='Outliers Removed')
        plot_prediction(kc_denoised, 'Kc', title='Noise Removed')
        plot_prediction(kc_filtered, 'Kc', title='Filtered by SWC')
    
    make_plot(kc_filtered, predictions=False)
    make_plot(kc_filtered)
    make_plot(kc_filtered, measures=False)
    
    return None    

@click.command()
@click.option('-v', '--visualize', is_flag=True)
def polish_kc(visualize):
    main(visualize)

if __name__ == "__main__":
    main(visualize=False)