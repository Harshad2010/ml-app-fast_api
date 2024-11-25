from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# Load the pretrained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
