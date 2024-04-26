# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:00:35 2022

@author: Federico Amato

Postrocess:
    - Rapporto
    - Outliers
    - Decomposizione stagionale
    - Filtro con soglia SWC < 0.21
Valutazione:
    - Curve trapezoidali
    - Tabella con le medie
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging

try:
    from predict import plot_prediction
except ModuleNotFoundError:
    from .predict import plot_prediction

from sklearn.ensemble import IsolationForest

from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn as sns

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def remove_outliers(df, detector):
    measured = df.loc[df["Source"] == "Measured"]
    detector.fit(measured["Kc"].values.reshape(-1, 1))
    inliers = detector.predict(df["Kc"].values.reshape(-1, 1))
    inliers = df.loc[inliers == 1]
    logging.info(f"Removed {len(df)-len(inliers)} outliers")
    return inliers


def remove_noise(df):
    decomposition = seasonal_decompose(
        df["Kc"], model="additive", period=365, extrapolate_trend=3
    )  #!!!
    mean_trend = decomposition.trend.mean()
    df_denoised = (decomposition.seasonal + mean_trend).to_frame(name="Kc")
    df_denoised["Source"] = df["Source"]
    return df_denoised


def swc_filter(df):
    try:
        swc = pd.read_pickle(ROOT_DIR / "data/interim" / "data.pickle")["SWC"]
        return df.loc[swc > 0.21]
    except KeyError:
        logging.warning("SWC not present in data, cannot use it as filter")

    # If SWC filter fails, return the original dataframe
    return df


def rolling_analysis(df):
    df_rolling = df["Kc"].rolling(window=30).mean().to_frame()
    df_rolling["Source"] = df["Source"]
    return df_rolling


def get_trapezoidal():
    return pd.read_csv(
        ROOT_DIR / "data/external" / "trapezoidal_kc.csv",
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
        dayfirst=True,
        skiprows=[0],
    )


def save_data(df, filename):
    df.to_csv(ROOT_DIR / "data/predicted" / f"{filename}.csv")
    df.to_pickle(ROOT_DIR / "data/predicted" / f"{filename}.pickle")
    return None


def make_trapezoidal(df):
    """
    Mid Season: 1Mag - 31Ago
    Final Season: 1Nov - 31Dec
    Initial Season: 1Gen - 31Mar
    """
    # Define the seasons
    seasons = {
        "Mid": set(range(5, 9)),  # May to August
        "Extreme": set(list(range(1, 4)) + [10, 11, 12]),  # October to March
    }

    # Function to apply to every day in the index to determine the season
    def get_season(date):
        for season, months in seasons.items():
            if date.month in months:
                return f'{season}{date.year}'
        return None

    # Apply the function to the DataFrame
    df["season"] = df.index.map(get_season)

    trapezoidal = df.groupby("season").mean(numeric_only=True)
    std = df.groupby("season").std(numeric_only=True)
    df["trapezoidal"] = df["season"].map(trapezoidal.to_dict()['Kc'])
    df["std"] = df["season"].map(std.to_dict()['Kc'])

    trpz = df.loc[:, ["trapezoidal", "std"]]
    trpz.to_pickle(ROOT_DIR / "data/predicted" / "trapezoidal.pickle")
    trpz.to_csv(ROOT_DIR / "data/predicted" / "trapezoidal.csv")
    return trpz


def add_plot_trapezoidal(ax, measures=None):
    trapezoidal = get_trapezoidal().loc["2018-01":]
    ax.plot(
        trapezoidal.iloc[:, 0],
        ls="--",
        c="blue",
        alpha=0.8,
        label=trapezoidal.columns[0],
    )
    ax.plot(
        trapezoidal.iloc[:, 1],
        ls=":",
        c="blue",
        alpha=0.8,
        label=trapezoidal.columns[1],
    )
    if measures is not None:
        ax.plot(measures["trapezoidal"].dropna(),
                ls="-.", c="green", label="Computed Trapezoidal", )
        error_up = measures["trapezoidal"] + measures["std"]
        error_down = measures["trapezoidal"] - measures["std"]
        ax.fill_between(error_up.index, error_up, error_down,
                        color="green", alpha=0.2)
    return ax


def add_plot_measures(df, ax):
    x = df.loc[df["Source"] == "Measured", "Kc"].index
    y = df.loc[df["Source"] == "Measured", "Kc"].values
    ax.scatter(x, y, s=5, label="Measured", c="r")
    return ax


def add_plot_predictions(df, ax):
    x = df.loc[df["Source"] == "Predicted", "Kc"].index
    y = df.loc[df["Source"] == "Predicted", "Kc"].values
    ax.scatter(x, y, s=5, label="Predicted", c="k")
    return ax


def add_plot_sma(df, ax):
    x = df.index
    y = df["Kc"].rolling(30).mean().values
    ax.plot(x, y, c="g", label="Simple Moving Average")
    return ax


# %%
def make_plot(*frames, trapezoidal=True, measures=True, sma=False, predictions=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Kc Predictions")
    df = frames[0]
    if trapezoidal:
        if len(frames) > 1:
            ax = add_plot_trapezoidal(ax, measures=frames[1])
        else:
            ax = add_plot_trapezoidal(ax)
    if measures:
        ax = add_plot_measures(df, ax)
    if predictions:
        ax = add_plot_predictions(df, ax)
    if sma:
        ax = add_plot_sma(df, ax)

    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="both", linestyle="--")
    ax.xticks = df.index
    plt.show()
    return ax


# %%
def main(outfile, visualize, contamination=0.01, seed=352):
    logging.info(f'{"-"*5} POLISH KC {"-"*5}')
    kc = pd.read_pickle(ROOT_DIR / "data/predicted" / "predicted.pickle")

    # Outlier removal
    detector = IsolationForest(contamination=contamination, random_state=seed)
    kc_inlier = remove_outliers(kc, detector)
    # Seasonal decomposition: take seasonal and mean trend and remove noise
    kc_denoised = remove_noise(kc_inlier)
    # SWC filter: take data with SWC > 0.21
    kc_filtered = swc_filter(kc_denoised)

    save_data(kc_filtered, outfile)

    if visualize:
        plot_prediction(kc, "Kc", title="Raw Predicted Kc")
        plot_prediction(kc_inlier, "Kc", title="Outliers Removed")
        plot_prediction(kc_denoised, "Kc", title="Noise Removed")
        plot_prediction(kc_filtered, "Kc", title="Filtered by SWC")

    kc_trapezoidal = make_trapezoidal(kc_filtered.copy())

    make_plot(kc_filtered, predictions=False)
    make_plot(kc_filtered, kc_trapezoidal)
    # make_plot(kc_filtered, measures=False)

    print(f'\n\n{"-"*21}')

    return kc_filtered


if __name__ == "__main__":
    output_name = "RF_kc_postprocessed" # Use the model name .upper()
    main(seed=352, contamination=0.01, visualize=True, outfile=output_name)
