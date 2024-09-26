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

import tomli
from sklearn.linear_model import LinearRegression

ROOT_DIR = Path(__file__).parent.parent.parent


def ask_for_model():
    saved_models = list(ROOT_DIR.glob("models/*"))
    print(f"You have {len(saved_models)} saved:")
    for m in saved_models:
        print(m)
    model_name = input("Which model do you want to use? ")
    return str(model_name)


def load_model(model_path):
    if model_path is None:
        model_path = ask_for_model()
    return joblib.load(model_path)


def fill_eta(eta, measured):
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


def rescale_eta(eta, scaler, input_folder, index=None, numpy=False):
    # Create fake DataFrame with fake features
    df = pd.read_pickle(input_folder / "preprocessed.pickle")
    if index is not None:
        df = df.loc[index]
    try:
        df["ETa"] = eta["ETa"]
    except IndexError:
        df["ETa"] = eta
    rescaled_eta = scaler.inverse_transform(df)[:, [-1]].ravel()
    if not numpy and index is not None:
        # Create a DataFrame
        rescaled_eta = pd.DataFrame(rescaled_eta, columns=["ETa"], index=index)
        rescaled_eta["Source"] = eta["Source"]
    return rescaled_eta


def find_eto(root_folder=ROOT_DIR):
    raw_data_path = root_folder / "data/raw"
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


def make_eto(eta_index, root_folder):
    """Impute missing ETo values using Iterative Imputer."""
    # Explicitly require this experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa

    # now you can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer

    eto = find_eto(root_folder)
    eto.index = eta_index
    # Impute missing features
    imputer = IterativeImputer(random_state=0)
    eto["ETo"] = imputer.fit_transform(eto)
    return eto


def rescale_series(eta, scaler, input_folder):
    # Reset original DataFrame with feature measures and predicted target
    df = pd.read_pickle(input_folder / "preprocessed.pickle")
    try:
        df["ETa"] = eta["ETa"]
    except IndexError:
        df["ETa"] = eta
    rescaled_df = scaler.inverse_transform(df)
    df = pd.DataFrame(rescaled_df, columns=df.columns, index=df.index)
    eta["ETa"] = df["ETa"].to_frame()
    if "ETo" not in df.columns:
        return eta
    eto = df["ETo"].to_frame()
    return eta, eto


def plot_prediction(df, series_name, title=None, ax=None):
    g = sns.scatterplot(data=df,
        x="Day",
        y=series_name,
        hue="Source",
        # height=6,
        # aspect=1.4,
        ax=ax
    )
    if title is not None and ax is not None:
        ax.set_title(title)
    return ax


def plot_linear(y_measured, y_predicted, features):
    max_vertex = 1.1 * max(y_measured.max(), y_predicted.max())

    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.scatter(y_measured, y_predicted, c="k", s=5, alpha=0.5)
    ax.plot([0, max_vertex], [0, max_vertex], "r--")
    ax.set_xlim(0, max_vertex)
    ax.set_ylim(0, max_vertex)
    ax.set_xlabel("Observed ETa [mm/day]")
    ax.set_ylabel("Predicted ETa [mm/day]")
    # ax.grid(True)

    # Linear Regression
    reg = LinearRegression(fit_intercept=False).fit(y_measured.reshape(-1, 1), y_predicted)
    ax.plot([0, max_vertex], [0, reg.coef_[0] * max_vertex], "k-", alpha=0.5, label=f"y={reg.coef_[0]:.2f}x")
    ax.legend()

    return fig


def compute_kc(eta, eto):
    """Compute Kc as ETa / ETo"""
    kc = pd.DataFrame()
    kc["Kc"] = eta["ETa"] / eto["ETo"]
    kc["Source"] = eta["Source"]
    return kc


def main(
    model_name, input_path=None, output_path=None, root_folder = ROOT_DIR,
    features=None, visualize=True, scaled=False,
):
    logging.info(f"\n{'-'*7} PREDICT ETa {'-'*7}\n\n")

    if not isinstance(root_folder, Path):
        root_folder = Path(root_folder)
    input_folder = root_folder / input_path
    output_folder = root_folder / output_path

    # Features to predict ETa
    X = pd.read_pickle(input_folder / "predict.pickle")
    if features is not None:
        X = X.loc[:, features]
    else:
        features = X.columns.tolist()
    measures = pd.read_pickle(input_folder / "preprocessed.pickle").dropna()

    # Scaler to rescale ETa
    scaler = joblib.load(root_folder / "models" / "scaler.joblib")
    
    # PREDICT ETA
    try:
        model = load_model(root_folder / "models" / f"{model_name}.joblib")
        logging.info(f"Predicting from features:\n" f"{X.columns.tolist()}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error finding the model. Remember to include file extension.") from e
    else:
        eta_predicted = model.predict(X)
        # Make a DataFrame of predictions
        eta = pd.DataFrame(eta_predicted, columns=["ETa Predicted"], index=X.index)
    # Make ETa DataFrame with measures and predictions
    eta = fill_eta(eta, measures)
    eta_rescaled = rescale_eta(eta, scaler, input_folder, index=eta.index)

    if visualize:
        if scaled:
            plot_prediction(eta, "ETa", "Measured and Predicted ETa (scaled)")
            plt.show()
            # plot_linear(y_measured, y_predicted, features)
        else:
            # Compute scaled predictions
            y_measured = eta_rescaled.loc[eta_rescaled["Source"] == "Measured", "ETa"].values
            X_measured = measures.loc[:, features]
            y_predicted = rescale_eta(model.predict(X_measured), scaler, input_folder, index=X_measured.index, numpy=True)
            plot_prediction(eta_rescaled, "ETa", "Measured and Predicted ETa")
            plt.show()
            linear_plot = plot_linear(y_measured, y_predicted, features)
            linear_plot.savefig(root_folder / "visualization/linear_plot.png")
            linear_plot.savefig(root_folder / "visualization/linear_plot.pdf")
            plt.show()
    # Save ETa
    if scaled:
        pd.to_pickle(eta, output_folder / "eta_predicted.pickle")
    else:
        pd.to_pickle(eta_rescaled, output_folder / "eta_predicted.pickle")
    logging.info(f"Predictions saved in:\n{output_folder / 'eta_predicted.pickle'}")

    # COMPUTE Kc AS ETa / ETo
    if "ETo" not in features:
        eto = make_eto(eta.index, root_folder)
        eta = rescale_series(eta, scaler, input_folder)
    else:
        eta, eto = rescale_series(eta, scaler, input_folder)
    kc = compute_kc(eta, eto)

    if visualize:
        plot_prediction(kc, "Kc", "Measured and Predicted Kc")
        plt.show()
    # Save Kc
    pd.to_pickle(kc, output_folder / "kc_predicted.pickle")
    logging.info(f"Predictions saved in:\n{output_folder}")
    logging.info(f'\n\n{"/"*30}\n\n')
    return kc


if __name__ == "__main__":
    # Read the configuration file available in a project directory
    project_dir = ROOT_DIR / "data/us_arm_fede"
    print(f"Reading configuration file from {project_dir}")
    with open(project_dir / "config.toml", "rb") as f:
        config = tomli.load(f)

    input_path = config["prediction"]["input_path"]
    output_path = config["prediction"]["output_path"]
    root_folder = project_dir
    features = config["preprocess"]["features"]
    visualize = config["prediction"]["visualize"]
    scaled = config["prediction"].get("scaled", True)

    for model_name in config["models"].keys():
        main(model_name, input_path, output_path, root_folder, features, visualize, scaled)
