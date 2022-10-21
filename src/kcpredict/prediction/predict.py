# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 18:43:00 2022

@author: Federico Amato

Predict ETa with saved model.
From measured and predicted ETa computes KC dividing by measured ET0.

"""
import click
import matplotlib.pyplot as plt
import pandas as pd
import joblib  # To load model and scaler
import seaborn as sns

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.parent


def ask_for_model():
    saved_models = list(ROOT_DIR.glob('models/*'))
    print(f"You have {len(saved_models)} saved:")
    for m in saved_models:
        print(m)
    model_name = input("Which model do you want to use? ")
    return str(model_name)


def load_model(model_name):
    if model_name is None:        
        model_name = ask_for_model()
    model = joblib.load(ROOT_DIR/'models'/f'{model_name}')
    return model


def fill_eta(eta):
    measured = pd.read_pickle(ROOT_DIR/'data/processed'/'processed.pickle')
    eta = pd.concat([eta, measured['ETa']], axis=1)
    eta.rename(columns={'ETa':'ETa Measured'}, inplace=True)
    idx_predict = eta['ETa Predicted'].dropna().index
    # Combine series
    total_eta = pd.DataFrame()
    total_eta['ETa'] = eta.iloc[:, 0].fillna(eta.iloc[:, 1])
    total_eta['Source'] = ['Predicted' if idx in idx_predict else 'Measured'
                     for idx in eta.index]
    return total_eta


def rescale_series(eta): 
    # Reset original DataFrame with feature measures and predicted target
    df = pd.read_pickle(ROOT_DIR/'data/processed'/'processed.pickle')
    df['ETa'] = eta['ETa']
    scaler = joblib.load(ROOT_DIR/'models'/'scaler.joblib')
    rescaled_df = scaler.inverse_transform(df)
    df = pd.DataFrame(rescaled_df, columns=df.columns, index=df.index)
    eta['ETa'] = df['ETa'].to_frame()
    eto = df['ETo'].to_frame()
    return eta, eto


def plot_prediction(df, series_name, title=None):
    g = sns.relplot(
        data=df,
        x='Day',
        y=series_name,
        hue='Source',
        height=6,
        aspect=1.6,
        )
    if title is not None:
        g.fig.suptitle(title)
    plt.show()


def compute_kc(eta):
    eta, eto = rescale_series(eta)
    kc = pd.DataFrame()
    kc['Kc'] = eta['ETa'] / eto['ETo']
    kc['Source'] = eta['Source']
    return kc


def main(model, output, visualize):
    print(f"\n\n{'-'*5} PREDICT ETa {'-'*5}\n\n")
    # Features to predict ETa
    X = pd.read_pickle(ROOT_DIR/'data/processed'/'predict.pickle').iloc[:, :-1]
    # Predict ETa
    try:
        model = load_model(model)
        print(f'Predicting from features:\n'
              f'{X.columns.tolist()}')
        eta = model.predict(X)
        # Make a DataFrame of predictions
        eta = pd.DataFrame(eta, columns=['ETa Predicted'], index=X.index)
    except FileNotFoundError:
        print("Error finding the model. Remember to include file extension.")
    # Make ETa DataFrame with measures and predictions
    eta = fill_eta(eta)
    if visualize:
        plot_prediction(eta, 'ETa', 'Measured and Predicted ETa (scaled)')
    # Compute Kc as ETa / ETo        
    kc = compute_kc(eta)
    if visualize:
        plot_prediction(kc, 'Kc', 'Measured and Predicted Kc')
    # Save Kc
    pd.to_pickle(kc, output)
    print(f'Predictions saved in:\n{output}')
    print(f'\n\n{"-"*25}\n\n')
    return None


@click.command()
@click.option('-m', '--model', prompt="Which model do you want to use?", 
                help="Name of saved model")
@click.option('-out', '--output-file', 
              type=click.Path(),
              default=(ROOT_DIR/'data/predicted'/'predicted.pickle'),)
@click.option('-v', '--visualize', is_flag=True)
def predict(model, output_file, visualize):
    """
    Predict ETa with given model.
    From measured and predicted ETa computes KC dividing by measured ET0.
    """
    main(model, output_file, visualize)

if __name__ == "__main__":
    predict()



