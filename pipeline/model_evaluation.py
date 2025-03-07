import pandas as pd
import logging
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Create logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create evaluation results directory
results_dir = "evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# Logging Configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data():
    """Load test dataset for evaluation."""
    try:
        X_test = pd.read_csv("processed_data/X_test.csv")
        y_test = pd.read_csv("processed_data/y_test.csv")
        logger.info("Successfully loaded test data.")
        return X_test, y_test
    except FileNotFoundError as e:
        logger.error("Test data files not found: %s", e, exc_info=True)
        raise
    except Exception as e:
        logger.error("Error loading test data: %s", e, exc_info=True)
        raise


def load_model():
    """Load the trained model from disk."""
    try:
        model_path = "models/model.pkl"
        model = joblib.load(model_path)
        logger.info("Successfully loaded the trained model.")
        return model
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e, exc_info=True)
        raise
    except Exception as e:
        logger.error("Error loading model: %s", e, exc_info=True)
        raise


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and save results."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info("Model Accuracy: %.4f", accuracy)
        logger.info("\nClassification Report:\n%s", report)

        # Save evaluation results
        results_path = os.path.join(results_dir, "evaluation.txt")
        with open(results_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        logger.info("Evaluation results saved successfully at %s", results_path)
    except Exception as e:
        logger.error("Error during model evaluation: %s", e, exc_info=True)
        raise


def main():
    try:
        X_test, y_test = load_data()
        model = load_model()
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        logger.error("Model evaluation process failed", exc_info=True)


if __name__ == "__main__":
    main()
