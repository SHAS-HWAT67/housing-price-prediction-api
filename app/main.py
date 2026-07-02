from fastapi import FastAPI
import joblib

from schema import House

app = FastAPI(
    title="House Price Prediction API"
)

model = joblib.load("../model/model.pkl")
scaler = joblib.load("../model/scaler.pkl")


@app.get("/")
def home():
    return {"message": "House Price Prediction API"}


@app.post("/predict")
def predict(data: House):

    values = [[
        data.area,
        data.bedrooms,
        data.bathrooms,
        data.stories,
        data.parking
    ]]

    values = scaler.transform(values)

    prediction = model.predict(values)

    return {
        "predicted_price": float(prediction[0])
    }