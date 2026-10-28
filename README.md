
# Food Delivery Time Prediction using Machine Learning

## Project Overview

This project predicts food delivery time using a machine learning model trained on food delivery data.

The model is trained in Google Colab and integrated into a FastAPI application for serving predictions.

## Technology Stack

* Python
* Pandas and NumPy
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Docker
* GitHub Actions

## Project Structure

```text
food_delivery_mlops/
├── models/
│   ├── food_delivery_model.joblib
│   └── food_delivery_scaler.joblib
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Features

* Food delivery time prediction
* Trained machine learning model
* Model preprocessing using a saved scaler
* REST API using FastAPI
* Interactive API documentation
* Docker containerization
* Automated testing and CI using GitHub Actions

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn main:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

## Docker

Build the image:

```bash
docker build -t food-delivery-mlops .
```

Run the container:

```bash
docker run -p 8000:8000 food-delivery-mlops
```

## Model Evaluation

The model is evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Actual evaluation results should be recorded after running the final training and testing pipeline.
