# 🏠 House Price Prediction API

A Machine Learning project that predicts house prices using Linear Regression and exposes predictions through a FastAPI REST API.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

## Features

- Train a Linear Regression model
- Predict house prices
- REST API with FastAPI
- Interactive API documentation using Swagger UI

## Run Project

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload