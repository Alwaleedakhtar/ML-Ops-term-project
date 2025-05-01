# MLOps Term Project – Loan Default Prediction

This project demonstrates an end-to-end machine learning lifecycle using MLflow. It focuses on predicting loan default in the finance domain and includes experiment tracking, model tuning, model registration, and performance monitoring.

## Project Structure

- `mlops_pipeline_local.py`: Main script to run baseline model, Hyperopt tuning, model registration, and performance drift simulation.
- `dataset_.csv`: The dataset used for model training and evaluation.
- `requirements.txt`: Python dependencies.

## How to Run

1. Clone the repo and navigate to the folder.
2. Start the MLflow server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

3. Run the pipeline script:

```bash
python mlops_pipeline_local.py
```

4. Open the MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) to track experiments and manage model registry.

## Features

- Logistic regression baseline
- Hyperparameter tuning with Hyperopt
- MLflow experiment tracking and model registry
- Performance monitoring with drift simulation

## Author

Al Valeed Akhtar (Bahçeşehir University)
