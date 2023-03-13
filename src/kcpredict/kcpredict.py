# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:51:27 2022

@author: Federico Amato

All the pipeline to predict Kc
"""
from data.make_data import main as make_data
from data.preprocess import main as preprocess_data
from models.make_model import main as make_model
from prediction.predict import main as predict
from prediction.polish import main as polish
from models.calc_metrics import main as calc_metrics

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent

# %% PARAMETERS

MAKE_DATA_PARAMETERS = {
    'input_file': 'G:/UNIPA/Dropbox/crop_coefficient/data/raw/data.xlsx',    
    'output_file': ROOT_DIR/'data/interim/data.pickle',
    'visualize': False,
    }

PREPROCESS_PARAMETERS = {
    'input_file': ROOT_DIR/'data/interim/data.pickle',
    'scaler': 'MinMax',
    'folds': 5,
    'k_seed': 24,  # 24
    'output_file': ROOT_DIR/'data/processed/processed.pickle',
    'visualize': True,
    }

MODEL_PARAMETERS = {
    'rf': {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 12,
        'ccp_alpha': 0.0,
        },
    'mlp': {
        'hidden_layer_sizes': (100, 100, 100),
        'max_iter': 1000,
        'random_state': 32652,  # 32652
        },
    'knn': {
        'n_neighbors': 5,
        'weights': 'distance',
        },
    }

PREDICTION_PARAMETERS = {
    'output': ROOT_DIR/'data/predicted'/'predicted.pickle',
    'visualize': True,
    }

POSTPROCESS_PARAMETERS = {
    'contamination': 0.01,
    'seed': 352,
    'visualize': True,
    }

# %% MAIN
def main():

    make_data(**MAKE_DATA_PARAMETERS)
    preprocess_data(**PREPROCESS_PARAMETERS)
    
    model_name = 'rf'
    model = RandomForestRegressor(**MODEL_PARAMETERS[model_name])
    make_model(model, model_name, visualize_error=False)
    
    predict(model=f'{model_name}.joblib', **PREDICTION_PARAMETERS)
    output_name = 'kc_postprocessed'
    polish(outfile=output_name, **POSTPROCESS_PARAMETERS)
    
    calc_metrics(output_name)
        

# %% Entry point
if __name__ == "__main__":
    main()


