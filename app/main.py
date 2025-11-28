from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# FastAPI app
app = FastAPI(title="Student Performance Prediction API")

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "linear_regression_model.pkl")
model = joblib.load(MODEL_PATH)

# Request schema
class StudentData(BaseModel):
    hours: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Student Performance Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict(data: StudentData):
    df = pd.DataFrame({"hours": [data.hours]})
    prediction = model.predict(df)
    return {"predicted_score": float(prediction[0])}
