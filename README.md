# End-to-End ML Pipeline using DVC & AWS S3 (Placement Dataset)

This project demonstrates an end-to-end machine learning pipeline using **DVC (Data Version Control)** and **AWS S3** for model management and reproducibility. The dataset used is a **placement dataset** for predicting student placement outcomes.

## 📌 Features
- **Data versioning** with DVC  
- **Pipeline automation** using `dvc.yaml`  
- **Model tracking** and **storage** in AWS S3  
- **ML workflow**: Data integration → Processing → Model Training → Evaluation → Visualization  

## 🛠 Tech Stack
- **Python** (Pandas, Scikit-learn, Matplotlib)  
- **DVC** for pipeline versioning  
- **AWS S3** for model storage  
- **Git & GitHub** for version control  

---

## 🚀 Pipeline Overview
The ML pipeline consists of four main stages:

1. **Data Integration (`data_integration.py`)**  
   - Reads and processes the placement dataset.  
   - Saves the cleaned data as `df.csv`.  

2. **Data Processing (`data_processing.py`)**  
   - Splits the dataset into training and test sets.  
   - Saves `X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv`.  

3. **Model Training (`model_building.py`)**  
   - Trains a Decision Tree classifier using **hyperparameters from `params.yaml`**.  
   - Saves the trained model as `model.pkl`.  

4. **Model Evaluation (`model_evaluation.py`)**  
   - Evaluates the model using accuracy and classification report.  
   - Logs the results.  

5. **Model Visualization (`model_visualization.py`)**  
   - Generates a decision tree plot (`decision_tree.png`).  

---

## 📂 Folder Structure
