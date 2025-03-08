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

## 🔧 Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/End-to-End-ML-Pipeline-using-DVC-AWS-S3.git
   cd End-to-End-ML-Pipeline-using-DVC-AWS-S3
   ```
2. **Create a virtual environment & Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (For Windows: venv\Scripts\activate)
   pip install -r requirements.txt
   ```
3. **Setup DVC & Initialize:**
   ```bash
   dvc init
   dvc remote add s3remote s3://your-bucket-name/path/
   ```
4. **Add & Push Data to S3 using DVC:**
   ```bash
   dvc add data/
   dvc push
   ```
5. **Run the ML Pipeline:**
   ```bash
   dvc repro
   ```

## 📸 Screenshots
### 🔹 IAM User Permissions

![Screenshot 2025-03-07 121707](https://github.com/user-attachments/assets/03cc4453-e767-4534-8b24-8f9761028276)

### 🔹 AWS S3 Bucket

![Screenshot 2025-03-07 121549](https://github.com/user-attachments/assets/b4ddec1f-e96a-40e4-bd15-89d9b73f259b)
## 📌 Results
- Achieved **65% accuracy** on the dataset
- Data is versioned and stored efficiently using **DVC & AWS S3**
- End-to-end automation of ML workflow

## 📜 License
This project is open-source and available under the **MIT License**.

## 🙌 Acknowledgments
Special thanks to the **Open-Source Community** for providing valuable resources on **MLOps & DVC**.


