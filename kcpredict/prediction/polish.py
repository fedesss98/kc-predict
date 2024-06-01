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


def swc_filter(df, root_folder=ROOT_DIR / "data/interim"):
    try:
        swc = pd.read_pickle(root_folder / "data/raw/data.pickle")["SWC"]
        return df.loc[swc > 0.21]
    except KeyError:
        logging.warning("SWC not present in data, cannot use it as filter")

    # If SWC filter fails, return the original dataframe
    return df


def rolling_analysis(df):
    df_rolling = df["Kc"].rolling(window=30).mean().to_frame()
    df_rolling["Source"] = df["Source"]
    return df_rolling


def get_trapezoidal(root_folder=ROOT_DIR / "data/external"):
    try:
        df = pd.read_csv(
            root_folder / "data/external/trapezoidal_kc.csv",
            sep=";",
            decimal=",",
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            dayfirst=True,
            skiprows=[0],
        )
    except FileNotFoundError:
        logging.warning("Trapezoidal file not found")
        return pd.DataFrame()
    return df


def save_data(df, output_folder, filename):
    df.to_csv(output_folder / f"{filename}.csv")
    df.to_pickle(output_folder / f"{filename}.pickle")
    return None


def read_allen(path):
    allen = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        parse_dates=True,
        infer_datetime_format=True,
        dayfirst=True,
    )
    try:
        allen = allen.loc[["Day", "Allen"]]
    except IndexError as e:
        raise IndexError(
            f"Allen Trapezoidal data must contain one column named Allen: {e}"
        )

    return allen


def extract_season_from_allen(allen):
    allen = allen.groupby("Allen")

    max_value = allen["Allen"].max()
    min_value = allen["Allen"].min()

    print(f"Max: {max_value}, Min: {min_value}")

    allen["Season"] = None

    for name, group in allen:
        start = group.index[0]
        end = group.index[-1]
        if max_value - 1e-2 <= name <= max_value:
            season = "High"
            print(f"{season} Kc season from {start} to {end}: {name}")
        elif min_value <= name <= min_value + 1e-2:
            season = "Low"
            print(f"{season} Kc season from {start} to {end}: {name}")
        else:
            season = "Mid"
        allen.loc[group.index, "Season1"] = season

    return allen


def make_trapezoidal(kc, allen, output_folder=ROOT_DIR / "data/predicted"):
    kc = kc.reset_index()
    # First recognize seasons based on Allen DataFrame
    allen_seasoned = extract_season_from_allen(allen).reset_index()

    allen_seasoned["month_day"] = allen_seasoned["Day"].dt.strftime("%m-%d")
    kc["month_day"] = allen_seasoned["Day"].dt.strftime("%m-%d")

    df = pd.merge(kc, allen_seasoned, on="month_day")
    df = df.sort_values("Day").drop(["month_day", "Date"], axis=1)
    df["year"] = df["Day"].dt.year
    df["Kc_trapezoidal"] = np.nan
    df["Error"] = np.nan
    groups = df.groupby(["year", "Season1"], group_keys=False)

    def _make_trapezoidal(group):
        season = group["Season"].iloc[0]
        if season != "Mid":
            group["Kc_trapezoidal"] = group["Kc"].mean()
            group["Error"] = group["Kc"].std()
        return group

    trapezoidal_df = groups.apply(_make_trapezoidal)
    # Fill Mid-season values with linear interpolation
    mid_values = trapezoidal_df["Kc_trapezoidal"].interpolate(method="linear")

    trpz = trapezoidal_df.loc[:, ["Kc_trapezoidal", "std"]]
    trpz.to_pickle(output_folder / "trapezoidal.pickle")
    trpz.to_csv(output_folder / "trapezoidal.csv")
    return trpz


def add_plot_trapezoidal(ax, measures=None):
    """
    Plot the Allen-Rallo Kc trapezoidal curves along with the trapezoidal computed from data
    """
    # trapezoidal = get_trapezoidal().loc["2018-01":]
    # ax.plot(trapezoidal.iloc[:, 0],
    #     ls="--", c="blue", alpha=0.8, label=trapezoidal.columns[0],)
    # ax.plot(trapezoidal.iloc[:, 1],
    #     ls=":", c="blue", alpha=0.8, label=trapezoidal.columns[1],
    # )
    if measures is not None:
        ax.plot(
            measures["trapezoidal"].dropna(),
            ls="-.",
            c="green",
            label="Computed Trapezoidal",
        )
        error_up = measures["trapezoidal"] + measures["std"]
        error_down = measures["trapezoidal"] - measures["std"]
        ax.fill_between(error_up.index, error_up, error_down, color="green", alpha=0.2)
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
def main(
    input_path,
    output_path,
    root_folder,
    visualize,
    trapezoidal_path=None,
    contamination=0.01,
    seed=352,
):
    logging.info(f'{"-"*5} POLISH KC {"-"*5}')

    if not isinstance(root_folder, Path):
        root_folder = Path(root_folder)
    input_folder = root_folder / input_path
    output_folder = root_folder / output_path

    kc = pd.read_pickle(input_folder / "kc_predicted.pickle")

    # Outlier removal
    detector = IsolationForest(contamination=contamination, random_state=seed)
    kc_inlier = remove_outliers(kc, detector)
    # Seasonal decomposition: take seasonal and mean trend and remove noise
    kc_denoised = remove_noise(kc_inlier)
    # SWC filter: take data with SWC > 0.21
    kc_filtered = swc_filter(kc_denoised, root_folder=root_folder)

    save_data(kc_filtered, output_folder, "kc_filtered")

    if visualize:
        plot_prediction(kc, "Kc", title="Raw Predicted Kc")
        plot_prediction(kc_inlier, "Kc", title="Outliers Removed")
        plot_prediction(kc_denoised, "Kc", title="Noise Removed")
        plot_prediction(kc_filtered, "Kc", title="Filtered by SWC")

    if trapezoidal_path is not None:
        allen = read_allen(trapezoidal_path)
        kc_trapezoidal = make_trapezoidal(kc_filtered.copy(), allen, output_folder)

    make_plot(kc_filtered, predictions=False)
    make_plot(kc_filtered, kc_trapezoidal)
    # make_plot(kc_filtered, measures=False)

    print(f'\n\n{"-"*21}')

    return kc_filtered


if __name__ == "__main__":
    output_name = "RF_kc_postprocessed"  # Use the model name .upper()
    main(seed=352, contamination=0.01, visualize=True, outfile=output_name)
