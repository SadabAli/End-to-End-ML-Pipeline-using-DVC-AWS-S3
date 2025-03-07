import pandas as pd
import logging
import yaml
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create models directory
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Logging Configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_building.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load model parameters from YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Loaded parameters from %s", params_path)
        return params["model"]
    except Exception as e:
        logger.error("Failed to load parameters: %s", e, exc_info=True)
        raise


def load_data() -> tuple:
    """Load processed training data."""
    try:
        X_train = pd.read_csv("processed_data/X_train.csv")
        y_train = pd.read_csv("processed_data/y_train.csv")
        X_test = pd.read_csv("processed_data/X_test.csv")
        y_test = pd.read_csv("processed_data/y_test.csv")

        logger.debug("Successfully loaded training and testing data")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logger.error("Missing processed data files: %s", e, exc_info=True)
        raise
    except Exception as e:
        logger.error("Error loading data: %s", e, exc_info=True)
        raise


def train_model(X_train, y_train, params):
    """Train a Decision Tree model."""
    try:
        model = DecisionTreeClassifier(
            criterion=params["criterion"],
            splitter=params["splitter"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
        )
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return model
    except Exception as e:
        logger.error("Model training failed: %s", e, exc_info=True)
        raise


def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info("Model Accuracy: %.4f", accuracy)
    except Exception as e:
        logger.error("Model evaluation failed: %s", e, exc_info=True)
        raise


def save_model(model):
    """Save trained model as a pickle file."""
    try:
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        logger.info("Model saved successfully at %s", model_path)
    except Exception as e:
        logger.error("Failed to save model: %s", e, exc_info=True)
        raise


def main():
    try:
        params = load_params("params.yaml")
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train, params)
        evaluate_model(model, X_test, y_test)
        save_model(model)
    except Exception as e:
        logger.error("Model building process failed", exc_info=True)


if __name__ == "__main__":
    main()
