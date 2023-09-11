# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 18:43:00 2022

@author: Federico Amato

Predict ETa with saved model.
From measured and predicted ETa computes KC dividing by measured ET0.

"""
import matplotlib.pyplot as plt
import pandas as pd
import joblib  # To load model and scaler
import seaborn as sns

import logging

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def ask_for_model():
    saved_models = list(ROOT_DIR.glob("models/*"))
    print(f"You have {len(saved_models)} saved:")
    for m in saved_models:
        print(m)
    model_name = input("Which model do you want to use? ")
    return str(model_name)


def load_model(model_name):
    if model_name is None:
        model_name = ask_for_model()
    return joblib.load(ROOT_DIR / "models" / f"{model_name}")


def fill_eta(eta):
    measured = pd.read_pickle(ROOT_DIR / "data/processed" / "processed.pickle")
    eta = pd.concat([eta, measured["ETa"]], axis=1)
    eta.rename(columns={"ETa": "ETa Measured"}, inplace=True)
    idx_predict = eta["ETa Predicted"].dropna().index
    # Combine series
    total_eta = pd.DataFrame()
    total_eta["ETa"] = eta.iloc[:, 0].fillna(eta.iloc[:, 1])
    total_eta["Source"] = [
        "Predicted" if idx in idx_predict else "Measured" for idx in eta.index
    ]
    return total_eta


def rescale_eta(eta, index=None):
    # Create fake DataFrame with fake features
    df = pd.read_pickle(ROOT_DIR / "data/processed" / "processed.pickle")
    df["ETa"] = eta["ETa"]
    scaler = joblib.load(ROOT_DIR / "models" / "scaler.joblib")
    rescaled_eta = scaler.inverse_transform(df)[:, [-1]].ravel()
    if index is not None:
        # Create a DataFrame
        rescaled_eta = pd.DataFrame(rescaled_eta, columns=["ETa"], index=index)
        rescaled_eta["Source"] = eta["Source"]
    return rescaled_eta


def find_eto():
    raw_data_path = ROOT_DIR / "data/raw"
    if (raw_data_path / "data.pickle").exists():
        eto = pd.read_pickle(raw_data_path / "data.pickle")["ETo"]
    elif (raw_data_path / "data.csv").exists():
        eto = pd.read_csv(
            raw_data_path / "data.csv", sep=";", decimal=",", usecols=["ETo"]
        )
    elif (raw_data_path / "data.xlsx").exists():
        eto = pd.read_excel(raw_data_path / "data.xlsx", usecols=["ETo"])
    else:
        eto_path = Path(input("Please input the position of ETo CSV file: "))
        eto = pd.read_csv(eto_path, sep=";", decimal=".", usecols=["ETo"])
    if isinstance(eto, pd.Series):
        # Convert it to dataframe
        eto = eto.to_frame(name="ETo")
    return eto


def make_eto(eta_index):
    from pathlib import Path

    # explicitly require this experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa

    # now you can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer

    eto = find_eto()
    eto.index = eta_index
    # Impute missing features
    imputer = IterativeImputer(random_state=0)
    eto["ETo"] = imputer.fit_transform(eto)
    return eto


def rescale_series(eta):
    # Reset original DataFrame with feature measures and predicted target
    df = pd.read_pickle(ROOT_DIR / "data/processed" / "processed.pickle")
    df["ETa"] = eta["ETa"]
    scaler = joblib.load(ROOT_DIR / "models" / "scaler.joblib")
    rescaled_df = scaler.inverse_transform(df)
    df = pd.DataFrame(rescaled_df, columns=df.columns, index=df.index)
    eta["ETa"] = df["ETa"].to_frame()
    try:
        eto = df["ETo"].to_frame()
    except KeyError:
        eto = make_eto(eta.index)
    return eta, eto


def plot_prediction(df, series_name, title=None):
    g = sns.relplot(
        data=df,
        x="Day",
        y=series_name,
        hue="Source",
        height=6,
        aspect=1.6,
    )
    if title is not None:
        g.fig.suptitle(title)
    plt.show()
    return None


def plot_linear(model, measures, features):
    X = measures.loc[:, features]
    y_measured = measures["ETa"].values
    y_predicted = model.predict(X)
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.scatter(y_measured, y_predicted, c="k")
    ax.plot([-1, 0, 1], [-1, 0, 1], "r--")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True)
    plt.show()
    return None


def compute_kc(eta):
    eta, eto = rescale_series(eta)
    kc = pd.DataFrame()
    kc["Kc"] = eta["ETa"] / eto["ETo"]
    kc["Source"] = eta["Source"]
    return kc


def main(
    model, output=None, features=None, visualize=True, scaled=True, eta_output=None
):
    logging.info(f"\n{'-'*7} PREDICT ETa {'-'*7}\n\n")
    # Features to predict ETa
    X = pd.read_pickle(ROOT_DIR / "data/processed" / "predict.pickle")
    if features is not None:
        X = X.loc[:, features]
    else:
        features = X.columns.tolist()
    measures = pd.read_pickle(ROOT_DIR / "data/processed" / "processed.pickle").dropna()
    # Predict ETa
    try:
        model = load_model(model)
        logging.info(f"Predicting from features:\n" f"{X.columns.tolist()}")
        eta_predicted = model.predict(X)
        # Make a DataFrame of predictions
        eta = pd.DataFrame(eta_predicted, columns=["ETa Predicted"], index=X.index)
    except FileNotFoundError:
        logging.error("Error finding the model. Remember to include file extension.")
    # Make ETa DataFrame with measures and predictions
    eta = fill_eta(eta)
    eta_rescaled = pd.DataFrame(columns=["ETa", "Source"])
    if visualize:
        if scaled:
            plot_prediction(eta, "ETa", "Measured and Predicted ETa (scaled)")
        else:
            eta_rescaled = rescale_eta(eta, index=eta.index)
            plot_prediction(eta_rescaled, "ETa", "Measured and Predicted ETa")
        plot_linear(model, measures, features)
    if eta_output is not None:
        # Save ETa
        if scaled:
            pd.to_pickle(eta, eta_output)
        else:
            pd.to_pickle(eta_rescaled, eta_output)
        logging.info(f"Predictions saved in:\n{eta_output}")
    elif output is not None:
        # Compute Kc as ETa / ETo
        kc = compute_kc(eta)
        if visualize:
            plot_prediction(kc, "Kc", "Measured and Predicted Kc")
        # Save Kc
        pd.to_pickle(kc, output)
        logging.info(f"Predictions saved in:\n{output}")
    logging.info(f'\n\n{"/"*30}\n\n')
    return kc if output is not None else None


if __name__ == "__main__":
    main()
