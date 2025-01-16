"""
Created by Federico Amato
16/01/2025

Look at the correlation of the noise extracted
from the Seasonal Decomposition with all the features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tomli
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Noise inspection")
parser.add_argument("-p", "--project", type=str, default="", help="Project folder name to inspect noise.")

def plot_scattermatrix(data):
    g = sns.relplot(data, x="value", y="Residual",
                    col="Feature", col_wrap=3,
                    facet_kws={"sharey":True, "sharex":False},
                    height=3, aspect=4/3)
    # Draw lines for the upper and right side of each axis
    for ax in g.axes.flat:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    # Adjust the titles of each subplot
    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), y=0.85)

    g.figure.subplots_adjust(wspace=0.03)

    plt.show()
    return g


def main(root, features_used):
    # Read noise data
    noise = pd.read_pickle(root / "data/postprocessed" / "kc_noise.pickle").to_frame().reset_index()
    noise.rename(columns={"resid": "Residual"}, inplace=True)

    # Read features
    features = pd.read_pickle(root / "data/raw/data.pickle")
    features = features.loc[:, features_used].reset_index()

    # Merge Dataframes
    df = features.merge(noise, on="Day")

    # Plot Correlations
    df_melted = df.melt(id_vars=["Residual"], var_name="Feature")  #Each feature become a category
    g = plot_scattermatrix(df_melted)

    # Save Plot
    g.savefig(root / "visualization" / "noise_correlations.png")

    return None


if __name__ == "__main__":

    args = parser.parse_args()
    
    if not args.project:
        config_folder = Path("./config.toml")
    else: 
        config_folder = Path(args.project) / "config.toml"

    # Load the config file
    with open(config_folder, "rb") as f:
        config = tomli.load(f)
            
    project_name = config["project_dir"]
    features = config["preprocess"]["features"]

    proj_root = Path(".") / project_name

    main(proj_root, features)
