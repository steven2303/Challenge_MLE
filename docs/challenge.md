# Flight Delay Prediction Model

## Part I: Model Implementation
### Overview

This repository contains the implementation of a logistic regression model for predicting flight delays. The model has been transcribed from a `.ipynb` notebook into a Python script (`model.py`) following best programming practices and ensuring it passes the required tests (`make model-test`).

### Features

- **Preprocessing pipeline**: Cleans and transforms raw flight data, handling categorical variables.
- **Logistic Regression Model**: A simple yet effective approach.
- **Class Imbalance Handling**: Implements class weighting to adjust for the disproportionate number of delayed vs. non-delayed flights.
- **Model Serialization**: Saves and loads trained models using `joblib`, eliminating the need for retraining.
- **Automated Testing**: Ensures correctness via `make model-test`.

### Key Implementations

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

1. **Simpler model with comparable performance to XGBoost**  
   - While a more complex model like XGBoost was tested, logistic regression **achieves very similar performance**, with differences only at the decimal level.  
   - Since **no specific metric was prioritized**, we consider the performance **optimal and comparable to XGBoost**, making logistic regression a viable choice.

2. **Lightweight Model for Production Efficiency**  
   - Logistic regression is significantly **lighter**, allowing for **faster inference times**.  
   - Unlike heavier models (e.g., XGBoost), it does not require an additional dependency on a larger library, **reducing the container size and overall resource consumption** in production.  
   - This is particularly important in **containerized deployments**, where minimizing image size and inference latency is crucial for scalability.


## Part II: API Deployment with FastAPI

### Overview

The trained flight delay prediction model has been deployed as an **API using FastAPI**. The `api.py` file provides a simple and efficient way to serve the model, enabling users to make predictions via **HTTP requests**.

### API Features

- **Health Check Endpoint (`/health`)**:  
  - Ensures that the API is running and operational.
  - Returns `{ "status": "OK" }` when the service is up.

- **Prediction Endpoint (`/predict`)**:  
  - Accepts a batch of flight data and returns a list of predicted delay probabilities.
  - Implements **input validation** using Pydantic to ensure data integrity.
  - Returns a JSON response with the predicted values.

- **Error Handling**:  
  - Handles **invalid requests** (e.g., missing or incorrect fields).
  - Returns meaningful **HTTP error responses** when validation fails.

In the test suite, there is a commented-out line that originally mocked the model prediction using mockito

```python
# when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
```
This was used to simulate model predictions in cases where the real model might not be available during testing.
However, since we serialize the trained model during make model-test, we can directly use the real trained model when testing the API. 

## Part III: Deploying the API to the Cloud

### Overview

The trained model and FastAPI service need to be deployed to a **cloud provider** to enable public or internal access for predictions. While **any cloud provider** can be used, we recommend **Google Cloud Platform (GCP)** for scalability and ease of deployment.

### Deployment Steps

1. **Dockerized API Deployment**  
   - A **Dockerfile** is included in the repository, containing all the necessary configurations to package and deploy the API.
   - The Docker image is built and pushed to **Artifact Registry** on GCP.

2. **Cloud Run Deployment**  
   - The Docker image is deployed from **Artifact Registry** to **Cloud Run**, ensuring a fully managed and scalable API service.
   - The API endpoint is automatically updated and exposed for external access.

3. **Update the API URL in the `Makefile`**  
   - Locate **line 26** in the `Makefile`.  
   - Replace the placeholder URL with the **publicly accessible URL** of your deployed API.

4. **Run Stress Tests**  
   - Ensure the API can handle concurrent requests by running:
     ```bash
     make stress-test
     ```
   - The **updated API endpoint** is already configured in the `stress-test` process to validate the production deployment.

### Cloud Deployment Workflow

- The **Dockerfile** ensures a consistent environment for running the FastAPI service.
- The **built image is stored in GCP's Artifact Registry** to maintain version control and security.
- **Cloud Run** provides a scalable and serverless execution environment.
- The **API endpoint is updated and validated** via automated stress tests.

## Part IV: Implementing CI/CD for the API Deployment

### Overview

A **CI/CD (Continuous Integration and Continuous Deployment) pipeline** has been implemented to automate the testing, building, and deployment process for the FastAPI service. The **CI pipeline (`ci.yml`) only includes the tests defined in the `Makefile`**, ensuring a streamlined and standardized validation process.

### CI Pipeline (`ci.yml`)

#### **Current Workflow in `ci.yml`**
The CI pipeline is triggered on **pushes and pull requests** to the `main`, `develop`, `deployment` branches. The workflow consists of the following steps:

1. **Check Out the Repository**  
   - Retrieves the latest code from GitHub.

2. **Set Up Python 3.12**  
   - Ensures a consistent runtime environment.

3. **Cache Virtual Environment**  
   - Speeds up dependency installation by caching Python dependencies.

4. **Create Virtual Environment & Install Dependencies**  
   - Runs `make venv` and `make install` only if dependencies are not cached.

5. **Run Model Tests** (`make model-test`)  
   - Ensures the model passes unit tests.

6. **Run API Tests** (`make api-test`)  
   - Verifies that the API functions correctly.

7. **Run Stress Tests (`make stress-test`) - Optional**
   - **Currently included but not essential in this phase** since the API endpoint should not be enabled until after deployment.
   - Stress testing is a **commodity process** that **should be executed post-deployment**, ensuring the API is live before performance validation.

8. **Upload Test Reports**  
   - Stores test results as artifacts for review.

### Next Steps: Container Build Validation

A future improvement to the CI/CD pipeline would be to **validate the container build within the CI stage** before triggering deployment. This would:

**Ensure that the Docker image builds successfully** before deployment.  
**Reduce deployment failures** caused by misconfigurations.  
**Improve efficiency** by catching issues earlier in the pipeline.  

### Deployment Workflow

1. **Trigger Deployment**  
   - The deployment is triggered when **CI tests pass successfully** on the `deployment` branch.

2. **Authentication with Google Cloud**  
   - The pipeline uses a **Service Account Key (`GCP_SA_KEY`)** stored in **GitHub Secrets** to authenticate to GCP.

3. **Enable Required GCP Services**  
   - Ensures that **Cloud Run** and **Artifact Registry** are enabled.

4. **Build and Push Docker Image**  
   - A **Dockerfile** is used to build the FastAPI service.
   - The image is **tagged and pushed** to **Artifact Registry**.

5. **Deploy to Cloud Run**  
   - The service is deployed to **Google Cloud Run** for **serverless execution**.
   - It is configured to **allow unauthenticated access** so that external requests can reach the API.

6. **Retrieve and Store the API Endpoint**  
   - The **Cloud Run URL** is extracted after deployment.
   - This URL is needed for **updating the `Makefile` and running stress tests**.

### Why This Approach?

✅ **Ensures Only Tested Code is Deployed**  
   - Deployment is triggered **only after CI tests pass**.

✅ **Uses Google Cloud's Best Practices**  
   - **Artifact Registry** for **secure Docker image storage**.
   - **Cloud Run** for **scalable and serverless API hosting**.

✅ **Automates Infrastructure Setup**  
   - The pipeline automatically **enables required GCP services**.
   - Ensures that **the Artifact Registry repository exists** before pushing images.
