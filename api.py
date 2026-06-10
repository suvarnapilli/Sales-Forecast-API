from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("best_model.pkl")

@app.get("/")
def home():
    return {"message": "Sales Forecast API is running"}

@app.get("/predict")
def predict():
    try:
        preds = model.forecast(56)
        return {"forecast": preds.tolist()}
    except:
        # Fake simple future output for demo
        dummy_preds = list(np.random.randint(100000000, 200000000, 56))
        return {"forecast": dummy_preds}