from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Define paths
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/BackupCrop_Recommendation .csv'
OUTPUT_DIR = 'output'

# Load the scaler and label encoder
def load_model(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

# Load models
try:
    scaler = load_model('scaler.pkl')
    label_encoder = load_model('label_encoder.pkl')
    stacked_model = load_model(os.path.join(OUTPUT_DIR, 'stacked_model.pkl'))
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=str(e))

# Define input data model
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    ph: float

# Define response model
class CropPredictionResponse(BaseModel):
    predicted_crop: str

# Prediction endpoint
@app.post("/predict/", response_model=CropPredictionResponse)
def predict_crop(input_data: CropInput):
    try:
        # Prepare the input data for prediction
        input_features = np.array([[input_data.N, input_data.P, input_data.K,
                                     input_data.Temperature, input_data.Humidity, input_data.ph]])
        # Scale the input data
        input_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = stacked_model.predict(input_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)

        return CropPredictionResponse(predicted_crop=predicted_crop[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
