from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import Water


app = FastAPI(
    title="Water Potability Prediction ",
    description="Predicting water potability using machine learning",
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def index():
    return {"message": "Welcome to the Water Potability Prediction API"}

@app.post("/predict")
def model_predict(water: Water):
    sample = pd.DataFrame([water.dict().values()], columns=water.dict().keys()) 
    predicted = model.predict(sample)
    

    if predicted[0] == 1:
        return "The water is potable"
    else:
        return "The water is not potable"
    




