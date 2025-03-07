import pandas as pd
import logging
import yaml
import os
from sklearn.model_selection import train_test_split
from typing import Union, Tuple

# Create logs directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Create data directory
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# Logging Configuration
logger = logging.getLogger('data_processing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_processing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


def split(data_path: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets and saves them as CSV.
    
    Args:
        data_path (str or pd.DataFrame): Path to CSV file or DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    try:
        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
        elif isinstance(data_path, pd.DataFrame):
            df = data_path
            logger.debug('Using provided DataFrame')
        else:
            raise ValueError("data_path must be a file path (str) or a DataFrame")

        # Splitting features and target
        X = df.drop(columns=['Survived'])
        y = df['Survived']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save the split data
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        logger.debug("Data successfully split and saved into train-test sets.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error("Error while splitting and saving data", exc_info=True)
        raise


def main():
    try:
        data_path = r"dataset/titanic_toy.csv"
        X_train, X_test, y_train, y_test = split(data_path)
        logger.info("Data split and saved successfully.")
    except Exception as e:
        logger.error("Failed to load, split, and save the dataset", exc_info=True)


if __name__ == "__main__":
    main()
