import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

try:
    from predict import plot_prediction
except ModuleNotFoundError:
    from .predict import plot_prediction

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


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


def read_allen(path):
    allen = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        parse_dates=True,
        infer_datetime_format=True,
        dayfirst=True,
        index_col=0,
    )
    try:
        allen = allen.loc[:, "Allen"]
    except IndexError as e:
        raise IndexError(
            f"Allen Trapezoidal data must contain one column named 'Allen': {e}"
        ) from e

    return allen


def extract_season_from_allen(allen):
    allen_groups = allen.groupby("Allen")

    max_value = allen["Allen"].max()
    min_value = allen["Allen"].min()

    print(f"Max: {max_value}, Min: {min_value}")

    allen["Season"] = None

    for name, group in allen_groups:
        start = allen.loc[group.index[0], 'Day'].dt.strftime("%m-%d")
        end = allen.loc[group.index[-1], 'Day'].dt.strftime("%m-%d")
        if max_value - 1e-2 <= name <= max_value:
            season = "High"
            print(f"{season} Kc season from {start} to {end}: {name}")
        elif min_value <= name <= min_value + 1e-2:
            season = "Low"
            print(f"{season} Kc season from {start} to {end}: {name}")
        else:
            season = "Mid"
        allen.loc[group.index, "Season"] = season

    return allen


def make_trapezoidal(kc, allen, output_folder=ROOT_DIR / "data/predicted"):
    kc = kc.reset_index()
    # First recognize seasons based on Allen DataFrame
    allen_seasoned = extract_season_from_allen(allen.reset_index())

    allen_seasoned["month_day"] = allen_seasoned["Day"].dt.strftime("%m-%d")
    kc["month_day"] = kc["Day"].dt.strftime("%m-%d")

    df = pd.merge(kc, allen_seasoned, on="month_day", suffixes=(None, "_Allen"))
    df = df.sort_values("Day").drop(["month_day", "Day_Allen"], axis=1)
    df["year"] = df["Day"].dt.year
    df["Kc_trapezoidal"] = np.nan
    df["Error"] = np.nan
    groups = df.groupby(["year", "Season"], group_keys=False)

    def _make_trapezoidal(group):
        season = group["Season"].iloc[0]
        if season != "Mid":
            group["Kc_trapezoidal"] = group["Kc"].mean()
            group["Error"] = group["Kc"].std()
        return group

    trapezoidal_df = groups.apply(_make_trapezoidal)
    # Fill Mid-season values with linear interpolation
    trapezoidal_df["Kc_trapezoidal"] = trapezoidal_df["Kc_trapezoidal"].interpolate(
        method="linear"
    )

    return trapezoidal_df.drop("year", axis=1)


def add_plot_allen(df, ax):
    x = df["Day"]
    y = df["Allen"]
    ax.plot(x, y, c="blue", alpha=0.8, label="Allen")
    return ax


def add_plot_trapezoidal(df, ax):
    x = df["Day"]
    y_true = df.apply(lambda x: x['Kc_trapezoidal'] if x['Season'] != 'Mid' else np.nan, axis=1)
    y_interp = df.apply(lambda x: x['Kc_trapezoidal'] if x['Season'] == 'Mid' else np.nan, axis=1)
    ax.plot(x, y_true, c="green", label="Trapezoidal")
    ax.plot(x, y_interp, c="green", ls="--", label="Interpolated values")
    error_up = y_true + df["Error"]
    error_down = y_true - df["Error"]
    ax.fill_between(x, error_up, error_down, color="green", alpha=0.2)
    return ax


def add_plot_measures(df, ax):
    x = df.loc[df["Source"] == "Measured", "Day"]
    y = df.loc[df["Source"] == "Measured", "Kc"].values
    ax.scatter(x, y, s=5, label="Measured", c="r")
    return ax


def add_plot_predictions(df, ax):
    x = df.loc[df["Source"] == "Predicted", "Day"]
    y = df.loc[df["Source"] == "Predicted", "Kc"].values
    ax.scatter(x, y, s=5, label="Predicted", c="k")
    return ax


def make_plot(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Kc Trapezoidal Reconstruction")

    # Plot Allen Trapezoidal
    add_plot_allen(df, ax)
    # Plot Postprocessed Kc Trapezoidal
    add_plot_trapezoidal(df, ax)
    # Plot Measured Kc
    add_plot_measures(df, ax)
    # Plot Predicted Kc
    add_plot_predictions(df, ax)

    # ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="both", linestyle="--")
    ax.xticks = df["Day"]
    plt.show()
    return ax


def main(input_path, output_path, trapezoidal_path, root_folder=ROOT_DIR, visualize=False, **kwargs):
    logging.info(f'{"-"*5} MAKING TRAPEZOIDAL KC {"-"*5}')

    if not isinstance(root_folder, Path):
        root_folder = Path(root_folder)
    input_folder = root_folder / input_path
    output_folder = root_folder / output_path

    kc = pd.read_pickle(output_folder / "kc_filtered.pickle")
    allen = read_allen(trapezoidal_path)
    kc_trapezoidal = make_trapezoidal(kc, allen, output_folder)

    if visualize:
        make_plot(kc_trapezoidal)

    trpz = kc_trapezoidal.loc[:, ["Day", "Kc_trapezoidal", "Error"]]
    trpz.to_pickle(output_folder / "trapezoidal.pickle")
    trpz.to_csv(output_folder / "trapezoidal.csv")

    return trpz

if __name__ == "__main__":
    input_path = "data/postprocessed"
    output_path = "data/postprocessed"
    root_folder = ROOT_DIR / "data/us_arm_fede"
    trapezoidal_path = ROOT_DIR / "data/external/trapezoidal_us_arm.csv"
    visualize = True

    main(input_path, output_path, trapezoidal_path, root_folder, visualize)