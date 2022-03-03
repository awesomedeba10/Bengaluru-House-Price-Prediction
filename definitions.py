import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

FEATURE_COL_PATH = os.path.join(PROJECT_ROOT, 'models', 'bengaluru_home_prices_columns.json')

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'bengaluru_home_prices_model.pickle')

SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'bengaluru_home_prices_scaler.pickle')

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Bengaluru_House_Data.csv')