## Flight Delay Prediction Model

#### Part I
## Overview

This repository contains the implementation of a logistic regression model for predicting flight delays. The model has been transcribed from a `.ipynb` notebook into a Python script (`model.py`) following best programming practices and ensuring it passes the required tests (`make model-test`).

## Features

- **Preprocessing pipeline**: Cleans and transforms raw flight data, handling categorical variables.
- **Logistic Regression Model**: A simple yet effective approach.
- **Class Imbalance Handling**: Implements class weighting to adjust for the disproportionate number of delayed vs. non-delayed flights.
- **Model Serialization**: Saves and loads trained models using `joblib`, eliminating the need for retraining.
- **Automated Testing**: Ensures correctness via `make model-test`.

## Key Implementations

#### Data Preparation
- Converts categorical variables into one-hot encoding.
- Computes the time difference between scheduled and actual departure times.
- Defines a binary classification target (`delay` > 15 minutes).
- **Stores valid airlines from training data** using `_save_training_categories()`, ensuring that only known airlines are considered in preprocessing.

#### Training and Model Persistence
- **Fits a logistic regression model** with appropriate class weights to handle class imbalance.
- **Ensures reproducibility and efficiency by serializing the model**:
  - `save_model(filepath)`: Saves the trained model to disk, preventing the need for retraining.
  - `load_model(filepath)`: Loads a previously trained model, allowing direct inference without executing the training process again.
  
The idea behind **serializing the model** is that once it has been trained, **it does not need to be retrained every time**. Instead, it can be directly loaded and invoked for inference, making the pipeline **faster, more efficient, and suitable for production environments**.

#### Inference
- Loads the model (if not already loaded) and predicts delays for new flight data.

### Model Selection Justification

The logistic regression model was selected due to the following reasons:

1. **Comparable Performance to XGBoost**  
   - Although a more complex model like XGBoost was tested, the logistic regression model **achieves a very similar performance**, with differences únicamente en el orden de los decimales.  
   - Since **no specific metric was prioritized**, we consider the performance **optimal and comparable to XGBoost**, making logistic regression a viable choice.

2. **Lightweight Model for Production Efficiency**  
   - Logistic regression is significantly **lighter**, allowing for **faster inference times**.  
   - Unlike heavier models (e.g., XGBoost), it does not require an additional dependency on a larger library, **reducing the container size and overall resource consumption** in production.  
   - This is particularly important in **containerized deployments** where minimizing image size and inference latency is critical for scalability.


### Usage

#### 1. Install dependencies
Ensure you have the required libraries installed:
```bash
pip install -r requirements.txt