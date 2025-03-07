import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
import yaml 
import os 
from typing import Union,Tuple 
from sklearn.metrics import accuracy_score,classification_report 
import logging 
import pickle

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler) 

def load_data(data_path: Union[str,pd.DataFrame]) -> pd.DataFrame:
    """Load data from a CSV file or use an existing DataFrame."""
    if isinstance(data_path, str):  # If data_path is a file path
        try:
            df = pd.read_csv(data_path)
            logger.debug('Data loaded from %s', data_path)
            return df
        except pd.errors.ParserError as e:
            logger.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            logger.error('Unexpected error occurred while loading the data: %s', e)
            raise
    elif isinstance(data_path, pd.DataFrame):  # If data_path is already a DataFrame
        logger.debug('Using provided DataFrame')
        return data_path
    else:
        raise ValueError("data_path must be a file path (str) or a DataFrame")
    
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> DecisionTreeClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained Decisiontree
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf = DecisionTreeClassifier()
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Load parameters (use default if params.yaml is missing)
        # params = load_params('params.yaml')
        
        # Load training data
        train_data = load_data(r'C:\Users\alisa\OneDrive\Desktop\MLOPs Note\data\raw\train.csv')
        X_train = train_data.iloc[:, :-1].values  # Features (all columns except the last)
        y_train = train_data.iloc[:, -1].values   # Labels (last column)
        
        # Train the model
        clf = train_model(X_train, y_train, params)
        
        # Save the trained model
        model_save_path = os.path.join('models', 'model.pkl')
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()