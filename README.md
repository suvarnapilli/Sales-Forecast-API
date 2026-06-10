# Sales Forecasting System

## Objective
Built an end-to-end time series forecasting system to predict next 8 weeks of sales using historical data.

## Features
- Handles missing dates and missing values
- Feature engineering with lag features and rolling averages
- Multiple forecasting models
- Automatic best model selection
- REST API using FastAPI

## Models Implemented
1. SARIMA
2. XGBoost

## Feature Engineering
- Lag Features:
  - lag_1
  - lag_7
  - lag_30
- Rolling Mean
- Day of Week
- Month

## Train-Test Split
Time-series based split without data leakage.

## Evaluation Metric
Mean Absolute Error (MAE)

## Best Model
XGBoost achieved the best MAE score.

## API Endpoints

### Home Endpoint
GET /

Returns API running status.

### Prediction Endpoint
GET /predict

Returns future sales forecasts.

## How to Run

### Install Libraries
```bash
pip install pandas numpy scikit-learn xgboost statsmodels fastapi uvicorn openpyxl joblib
```

### Run Training
```bash
python main.py
```

### Run API
```bash
uvicorn api:app --reload
```

### Open API Docs
```text
http://127.0.0.1:8000/docs
```

## Files
- main.py → training pipeline
- api.py → REST API
- best_model.pkl → saved model
- data.xlsx → dataset
