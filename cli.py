"""
@author: Federico Amato

Command line Interface to predict Crop Coefficient Kc
"""

import click
from kcpredict.data.make_data import main as make_data_main
from kcpredict.data.preprocess import main as preprocess_data_main
from kcpredict.models.make_model import make_model
from kcpredict.prediction.predict import main as predict_main
from kcpredict.prediction.polish import main as polish_main

from pathlib import Path
ROOT_DIR = Path(__file__)

@click.group()
def cli():
    """
    \n\n
    *****************************************************
    \n
    Command Line Interface to predict Crop Coefficient Kc 
    via Machine Learning Models.
    
    Run commands in order first time:\n
    \t- make-data\n
    \t- preprocess-data\n
    \t- make-model\n
    \t- predict\n
    *****************************************************
    """
    return None


@click.group(chain=True)
def make_all():
    """
    Go through all the pipeline
    """
    return None
cli.add_command(make_all)


# Functions
@cli.command()
@click.option('-in', '--input-file', type=click.Path(exists=True), default=(ROOT_DIR/'data/raw'/'data.xlsx'))
@click.option('-out', '--output-file', type=click.Path(), default=(ROOT_DIR/'data/interim'/'data.pickle'))
@click.option('-v', '--visualize', is_flag=True,)
def make_data(input_file, output_file, visualize):
    """
    Read raw CSV file and save the dataframe in a Pickle file.
    """
    make_data_main(input_file, output_file, visualize=visualize)
    return None


@cli.command()
@click.option('-in', '--input-file', type=click.Path(), default=(ROOT_DIR /'data/interim'/'db_villabate.pickle'))
@click.option('-s', '--scaler', default='MinMax', type=click.Choice(['Standard', 'MinMax'], case_sensitive=False))
@click.option('-k', type=click.INT, default=5, help="Number of folds")
@click.option('--k-seed', type=click.INT, default=2, help="Number of folds")
@click.option('-out', '--output-file', type=click.Path(), default=(ROOT_DIR/'data/processed'/'processed.pickle'))
@click.option('-v', '--visualize', is_flag=True,)
def preprocess_data(input_file, scaler, k, k_seed, output_file, visualize):
    """
    Preprocess ETa data for predictions.

    - Select features
    - Impute missing values with KNNImputer
    - Split data with KFolds
    - Scale data with StandardScaler or MinMaxScaler

    Train set must never see test set, even during the scaling.
    Therefore K-folds slpit must come before the scaling.

    """
    preprocess_data_main(input_file, scaler, k, k_seed, output_file, visualize=visualize)


#This is a command group better defined in its own file
cli.add_command(make_model)


@cli.command()
@click.option('-m', '--model', prompt="Which model do you want to use?", help="Name of saved model")
@click.option('-out', '--output-file', type=click.Path(), default=(ROOT_DIR/'data/predicted'/'predicted.pickle'),)
@click.option('-v', '--visualize', is_flag=True)
def predict(model, output_file, visualize):
    """
    Predict ETa with given model.
    From measured and predicted ETa computes KC dividing by measured ET0.
    """
    predict_main(model, output_file, visualize)



@cli.command()
@click.option('-v', '--visualize', is_flag=True)
def polish_kc(visualize):
    polish_main(visualize)


# make_all.add_command(make_data)
# make_all.add_command(preprocess_data)
# make_all.add_command(make_model)
# make_all.add_command(predict)



if __name__ == "__main__":
    cli()