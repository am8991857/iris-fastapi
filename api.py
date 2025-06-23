from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load model at startup
model = joblib.load("iris_model.pkl")

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"msg": "Iris model is running!"}

@app.post("/predict")
def predict(data: IrisInput):
    X = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = model.predict(X)[0]
    return {
        "class_index": int(pred),
        "class_name": ["setosa", "versicolor", "virginica"][pred]
    }
