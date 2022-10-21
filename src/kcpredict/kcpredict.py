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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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
    'k_seed': 5487,
    'output_file': ROOT_DIR/'data/processed/processed.pickle',
    'visualize': True,
    }

MODEL_PARAMETERS = {
    'n_estimators': 100,
    'random_state': 36485,
    }

PREDICTION_PARAMETERS = {
    'model': 'rf.joblib',
    'output': ROOT_DIR/'data/predicted'/'predicted.pickle',
    'visualize': True,
    }

POSTPROCESS_PARAMETERS = {
    'visualize': True,
    }

# %% MAIN
def main():

    make_data(**MAKE_DATA_PARAMETERS)
    preprocess_data(**PREPROCESS_PARAMETERS)
    
    model_name = 'rf'
    model = RandomForestRegressor(**MODEL_PARAMETERS)
    make_model(model, model_name)
    
    predict(**PREDICTION_PARAMETERS)
    polish(**POSTPROCESS_PARAMETERS)
        

# %% Entry point
if __name__ == "__main__":
    main()


