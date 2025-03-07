import pandas as pd
import os
import yaml
import logging
from typing import Union

# Creating logs directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_integration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handle_log_path = os.path.join(log_dir, 'data_integration.log')
file_handler = logging.FileHandler(file_handle_log_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Fixed typo
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Loaded parameters from %s", params_path)
        return params
    except Exception as e:
        logger.error("Failed to load parameters: %s", e, exc_info=True)
        raise


def load_data(data_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load the dataset from a CSV file or return an existing DataFrame.
    """
    if isinstance(data_path, str):
        try:
            df = pd.read_csv(data_path)
            logger.debug("Data loaded from %s", data_path)
            return df
        except Exception as e:
            logger.error("Unexpected error occurred while loading the data: %s", e, exc_info=True)
            raise
    elif isinstance(data_path, pd.DataFrame):
        logger.debug("Using provided DataFrame")
        return data_path
    else:
        raise ValueError("Data must be a file path (str) or a DataFrame")


def main():
    try:
        data_path = r"dataset\titanic_toy.csv"
        df = load_data(data_path)
        print(df)
    except Exception as e:
        logger.error('Failed to load the dataset: %s', e, exc_info=True)
        print(e)


if __name__ == "__main__":
    main()
