"""
Created by Federico Amato
16/01/2025

Look at the correlation and statistics of the noise extracted
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
parser.add_argument("-v", "--verbose", action="store_true", help="Whether or not to display plots.")


def plot_scattermatrix(data, display=True):
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

    if display:
        plt.show()
    
    return g


def get_statistics(data):
    minimum = data.min()
    maximum = data.max()
    mean = data.mean()
    median = data.median()
    std = data.std()
    var = data.var()  # Unbiased variance normalized by N-1
    statistics = pd.DataFrame(
            [minimum, maximum, mean, median, std, var],
            index = ["Min", "Max", "Mean", "Median", "Std", "Variance"],
            columns = ["Residual Statistics"]
        )
    stats_string = '\n'.join((
        f"Min: {minimum:.3f}",
        f"Max: {maximum:.3f}",
        f"Mean: {mean:.3f}",
        f"Median: {median:.3f}",
        f"Std: {std:.3f}",
        f"Variance: {var:.3f}",
    ))
    return stats_string, statistics


def plot_statistics(data, x=None, hue=None, statistics=None, display=True):
    if not x:
        g = sns.displot(data, kde=True, stat="density", legend=False)
    else:
        g = sns.displot(data, x=x, hue=hue, kde=True, stat="density", multiple="stack")
    
    g.axes[0, 0].set_title("Noise PDF", fontsize=16)

    # Draw lines for the upper and right side of each axis
    for ax in g.axes.flat:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    if statistics:
        boxprops = dict(ec='black', fc='white')
        g.figure.text(0.7, 0.9, statistics, 
                      fontsize=12, verticalalignment='top', bbox=boxprops)
    
    if display:   
        plt.show()

    return g


def main(root, features_used, verbose=False):
    # Read noise data
    noise = pd.read_pickle(root / "data/postprocessed" / "kc_noise.pickle").to_frame().reset_index()
    noise.rename(columns={"resid": "Residual"}, inplace=True)

    # Read features
    features = pd.read_pickle(root / "data/raw/data.pickle")
    features = features.loc[:, features_used + ['ETa']].reset_index()

    # Merge Dataframes
    df = features.merge(noise, on="Day")

    # Plot Correlations
    df_melted = df.melt(id_vars=["Residual"], var_name="Feature")  #Each feature become a category
    g = plot_scattermatrix(df_melted, display=verbose)

    # Save Plot
    g.savefig(root / "visualization" / "noise_correlations.png")
    g.savefig(root / "visualization" / "noise_correlations.eps")

    # Save Pearson Correlation Coefficient in a file
    p_corr = df.corr(numeric_only=True)["Residual"]
    print(p_corr)
    p_corr.to_csv(root / "data/postprocessed" / "kc_noise_correlations.csv")
    
    # Get various statistics of the noise
    statistics, csv = get_statistics(noise["Residual"])
    print()
    print(csv)
    csv.to_csv(root / "data/postprocessed" / "kc_noise_statistics.csv", header=False)

    # Plot Probability Distribution Function
    g_stats = plot_statistics(noise, statistics=statistics, display=verbose)
    g_stats.savefig(root / "visualization" / "noise_statistics.png")
    g_stats.savefig(root / "visualization" / "noise_statistics.eps")

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

    main(proj_root, features, args.verbose)
