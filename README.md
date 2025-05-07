# Loan Default Prediction with MLflow Lifecycle Management

This project demonstrates a complete MLOps pipeline for predicting loan defaults using three machine learning models: **Logistic Regression**, **Random Forest**, and **XGBoost**. The full ML lifecycle — including experiment tracking, hyperparameter tuning, model registry, and monitoring — is managed using **MLflow**.

## Project Structure

```
├── dataset_.csv                  # Preprocessed dataset used in Colab
├── MLOps.ipynb                   # Unified Google Colab notebook
├── requirements.txt              # Python environment dependencies
├── mlruns/                       # MLflow logs (stored in Google Drive or locally)
└── README.md                     # Project documentation
```

## Features

- **Experiment Tracking** with MLflow
- **Hyperparameter Tuning** via Hyperopt
- **Final Model Comparison** with Accuracy, Precision, Recall, and F1-score
- **Model Registry** and Artifact Logging
- **Data Drift Simulation** for Monitoring Robustness

## Models Implemented

| Model               | Accuracy | F1 Score | Precision | Recall |
|---------------------|----------|----------|-----------|--------|
| Logistic Regression | ~81.6%   | ~0.48    | ~0.49     | ~0.45  |
| Random Forest       | ~81.8%   | ~0.50    | ~0.51     | ~0.50  |
| XGBoost             | ~82.4%   | ~0.65    | ~0.49     | ~0.97  |

> **XGBoost** had the best performance, particularly in recall — crucial for minimizing missed defaulters.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/loan-default-mlops.git
cd loan-default-mlops
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Launch MLflow UI (Optional, local environment)
```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Reproducibility

- Code and dataset provided in a single Colab notebook
- MLflow tracking URI is set to log runs in Google Drive
- Exportable `mlruns/` folder for local MLflow comparison
- Requirements file included for consistent environment setup

## Visualization

Models were compared side-by-side in the MLflow UI using logged metrics and parameters. Parallel coordinates plots helped assess trade-offs between F1, recall, and precision.

## Notes

- Dataset used: anonymized loan application data
- Class imbalance handled with `class_weight='balanced'` and boosting
- All models were trained and evaluated under a unified preprocessing pipeline

## Author

**Al Valeed Akhtar**  
Bahçeşehir University – Artificial Intelligence Engineering
