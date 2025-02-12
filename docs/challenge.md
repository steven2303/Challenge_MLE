## Flight Delay Prediction Model

#### Part I
## Overview

This repository contains the implementation of a logistic regression model for predicting flight delays. The model has been transcribed from a `.ipynb` notebook into a Python script (`model.py`) following best programming practices and ensuring it passes the required tests (`make model-test`).

## Features

- **Preprocessing pipeline**: Cleans and transforms raw flight data, handling categorical variables.
- **Logistic Regression Model**: A simple yet effective approach.
- **Class Imbalance Handling**: Implements class weighting to adjust for the disproportionate number of delayed vs. non-delayed flights.
- **Model Persistence**: Supports saving and loading trained models using `joblib`.
- **Automated Testing**: Ensures correctness via `make model-test`.

## Key Implementations

#### Data Preparation
- Converts categorical variables into one-hot encoding.
- Computes the time difference between scheduled and actual departure times.
- Defines a binary classification target (`delay` > 15 minutes).
- **Stores valid airlines from training data** using `_save_training_categories()`, ensuring that only known airlines are considered in preprocessing.

#### Training and Model Persistence
- **Fits a logistic regression model** with appropriate class weights to handle class imbalance.
- **Ensures reproducibility by saving and loading the model**:
  - `save_model(filepath)`: Saves the trained model to disk, ensuring that future inference or retraining starts from a known state.
  - `load_model(filepath)`: Loads a previously trained model, allowing seamless deployment without the need for retraining.
  
These persistence functions ensure that the model remains **consistent across different runs** and **avoids unnecessary retraining**, which is critical for a scalable and reproducible workflow in production.

#### Inference
- Loads the model (if not already loaded) and predicts delays for new flight data.

### Model Selection Justification

The logistic regression model was selected due to its:
- **Interpretability**: Provides clear insights into the impact of different factors on flight delays.
- **Performance**: Balances simplicity and effectiveness for the given task.
- **Efficiency**: Fast training and inference times, making it suitable for real-time or batch processing scenarios.

### Usage

#### 1. Install dependencies
Ensure you have the required libraries installed:
```bash
pip install -r requirements.txt