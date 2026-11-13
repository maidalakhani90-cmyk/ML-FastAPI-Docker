
# Import required libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from pathlib import Path

# Create FastAPI application
app = FastAPI(
    title="Iris Flower Prediction API",
    description="Machine Learning API using FastAPI",
    version="1.0"
)

# Load trained machine learning model
MODEL_PATH = Path(__file__).parent / "model" / "iris_model.joblib"
model = joblib.load(MODEL_PATH)

# Flower input data
class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Iris Flower Prediction API",
        "status": "API is running"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: FlowerInput):

    # Prepare input data
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Make prediction
    prediction = model.predict(input_data)

    # Convert prediction into flower name
    flower_names = ["Setosa", "Versicolor", "Virginica"]
    result = flower_names[int(prediction[0])]

    return {
        "prediction": result
    }
