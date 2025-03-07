import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import plot_tree
import os

# Load model
model_path = "models/model.pkl"
model = joblib.load(model_path)

# Load test data
X_test = pd.read_csv("processed_data/X_test.csv")
y_test = pd.read_csv("processed_data/y_test.csv")

# Create output directory
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

#  1. **Decision Tree Plot (only if applicable)**
if hasattr(model, "tree_"):
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=X_test.columns, class_names=["Not Survived", "Survived"], filled=True)
    plt.savefig(f"{output_dir}/decision_tree.png", dpi=300)
    plt.close()
    print(" Decision Tree saved as decision_tree.png")
else:
    print(" Model is not a decision tree. Skipping decision tree plot.")

#  2. **Feature Importance (if applicable)**
if hasattr(model, "feature_importances_"):
    plt.figure(figsize=(10, 5))
    importance = pd.Series(model.feature_importances_, index=X_test.columns)
    importance.nlargest(10).plot(kind='barh', colormap="viridis")
    plt.title("Feature Importance")
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()
    print(" Feature importance saved as feature_importance.png")

#  3. **Confusion Matrix**
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()
print(" Confusion Matrix saved as confusion_matrix.png")

#  4. **ROC Curve**
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probability for class 1
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    print(" ROC Curve saved as roc_curve.png")
else:
    print(" Model does not support probability predictions. Skipping ROC Curve.")

print(" Model visualization completed! Check the 'visualizations' folder.")
