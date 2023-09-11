# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:30:10 2023

@author: Federico Amato

Compute various metrics comparing postprocessed KC with:
    - Measured KC
    - Allen
    - Rallo
    - VI model
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def get_trapezoidal():
    return pd.read_csv(
        ROOT / "data/external" / "trapezoidal_kc.csv",
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
        dayfirst=True,
        skiprows=[0],
    )


def get_vi_model():
    vi = pd.read_csv(ROOT / "data/external/VIs_Kc_2018_2022.csv", sep=";", decimal=",", index_col=0, parse_dates=True,
                      infer_datetime_format=True, dayfirst=True, )
    vi.columns = ["VI Kc"]
    return vi


def plot_models(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot(ax=ax)
    plt.show()
    return None


def save_all_models(df, measures, allen, rallo, vi):
    df = df.loc[:, (slice(None), 'Kc')]
    df.columns = [" ".join(col).rstrip("_") for col in df.columns.values]
    all_models = pd.concat([df, measures, allen, rallo, vi], axis=1)
    all_models.to_csv(ROOT / "data/predicted" / "all_models.csv", sep=";", decimal=",")
    all_models.to_pickle(ROOT / "data/predicted" / "all_models.pickle")
    return None


def calc_errors(true, predictions, metric_name):
    errors = []
    models = predictions.columns.get_level_values(0)
    for m in models:
        model_predictions = predictions.loc[:, (m, "Kc")].dropna()
        df = pd.concat([true, model_predictions], axis=1, join="inner")
        rmse = mean_squared_error(df.iloc[:, 0], df.iloc[:, 1], squared=False)
        print(f"{m} RMSE {metric_name}: {rmse:.4f}")
        errors.append(rmse)
    return errors


def main(file):
    print(f'{"-"*5} COMPARE MODELS TO CALC METRICS {"-"*5}\n\n')

    predictions = pd.read_pickle(ROOT / "data/predicted" / "predicted.pickle")
    if isinstance(file, str):
        processed_kc = pd.read_pickle(ROOT / f"data/predicted/{file}.pickle")["Kc"]
    elif isinstance(file, pd.DataFrame):
        processed_kc = file
    else:
        raise TypeError("file must be a string or a DataFrame")

    measures = predictions.loc[predictions["Source"] == "Measured"]["Kc"]
    measures.columns = ["Measured Kc"]
    theoretical = get_trapezoidal()
    allen = theoretical.iloc[:, 0].to_frame()
    rallo = theoretical.iloc[:, 1].to_frame()
    vi = get_vi_model()

    save_all_models(processed_kc, measures, allen, rallo, vi)

    calc_errors(measures, processed_kc, "Kc Postprocessed on Measures")
    calc_errors(allen, processed_kc, "Kc Postprocessed on Allen")
    calc_errors(rallo, processed_kc, "Kc Postprocessed on Rallo")
    calc_errors(vi, processed_kc, "Kc Postprocessed on Vi Kc")

    print(f'\n\n{"-"*21}')
    return None


if __name__ == "__main__":
    filename = "kc_postprocessed"
    main(filename)
