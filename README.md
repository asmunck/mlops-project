# NYC Taxi Duration Prediction - MLOps Project

## Overview

This project demonstrates a complete Machine Learning workflow for predicting the duration of taxi trips in New York City. It includes data processing, model training, evaluation, and deployment using modern MLOps practices.

**Note:** The data Parquet files exceed GitHub's size limits and are managed using DVC (Data Version Control).

## Project Structure

*   `data/`: Contains the raw data (Parquet files).  These files are tracked with DVC.
*   `models/`: Stores trained models (pickled files).
*   `training/`: Contains the training code (`duration-prediction.ipynb`).
*   `mlruns/`: MLflow tracking directory (stores experiment runs, parameters, metrics, and artifacts).
*   `README.md`: This file.

## Data

The dataset used in this project is the Green Taxi Trip Records from the NYC Taxi and Limousine Commission.  The data includes pickup and dropoff dates/times, locations, trip distances, and other relevant information.

*   `green_tripdata_2024-01.parquet`: Training data (January 2024).
*   `green_tripdata_2024-02.parquet`: Validation data (February 2024).

## Workflow

1.  **Data Ingestion and Preprocessing:**
    *   Data is read from Parquet files using `pandas`.
    *   Trip duration is calculated from pickup and dropoff times.
    *   Outliers in trip duration are filtered out (1 to 60 minutes).
    *   Categorical features (`PULocationID`, `DOLocationID`) are converted to strings.
    *   A combined `PU_DO` feature is created by concatenating pickup and dropoff location IDs.
2.  **Feature Engineering:**
    *   `DictVectorizer` from `scikit-learn` is used to convert categorical features into numerical features using one-hot encoding.
3.  **Model Training:**
    *   **Linear Regression:** A simple `LinearRegression` model is trained as a baseline.
    *   **Lasso Regression:** A `Lasso` model is trained with L1 regularization, and the results are tracked using MLflow.
    *   **XGBoost:** An XGBoost model is trained with hyperparameter optimization using Hyperopt.
4.  **Hyperparameter Optimization:**
    *   Hyperopt is used to search for the best hyperparameters for the XGBoost model.
    *   The `objective` function trains and evaluates the XGBoost model for a given set of hyperparameters.
    *   The search space includes parameters such as `max_depth`, `learning_rate`, `reg_alpha`, `reg_lambda`, and `min_child_weight`.
    *   The TPE (Tree-structured Parzen Estimator) algorithm is used to guide the search.
5.  **Experiment Tracking:**
    *   MLflow is used to track all experiments, including parameters, metrics (RMSE), and artifacts (trained models).
6.  **Model Persistence:**
    *   Trained models and `DictVectorizer` objects are saved using `pickle`.

## Code Overview

*   **`training/duration-prediction.ipynb`:**  Jupyter Notebook containing the complete training pipeline.

    *   **Dependencies:** `pandas`, `scikit-learn`, `mlflow`, `xgboost`, `hyperopt`.

## MLflow

MLflow is used for experiment tracking. To view the MLflow UI, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
