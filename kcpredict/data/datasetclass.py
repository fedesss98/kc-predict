"""
Class Dataset:
    - This class is used to load the dataset and preprocess it.
"""
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent

class Dataset:
    def __init__(self, name):
        self.path = ROOT / f"data/processed/{name}.pickle"
        self.dataset = None
        if not self.path.exists():
            raise FileNotFoundError(f"No Dataset named '{self.path}'. First preprocess the data.")


    def load(self):
        self.data = pd.read_csv(self.path)

    def save(self, name):
        self.dataset.to_pickle(name)