from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Define paths
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/BackupCrop_Recommendation.csv'
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
    model1 = load_model(os.path.join(OUTPUT_DIR, 'stacked_model.pkl'))
    
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
    model1_output: str
    model2_recommendation: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Prediction endpoint
@app.post("/predict/", response_model=CropPredictionResponse)
def predict_crop(input_data: CropInput):
    try:
        # Prepare the input data for prediction
        input_features = np.array([[input_data.N, input_data.P, input_data.K,
                                     input_data.Temperature, input_data.Humidity, input_data.ph]])
        # Scale the input data
        input_scaled = scaler.transform(input_features)

        # Model 1 Prediction
        prediction1 = model1.predict(input_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction1)[0]

        # Model 2 Prediction (Recommendation based on Model 1 output)
        prediction2_input = np.array([[predicted_crop]])  # Modify this line based on Model 2’s input format
        model2_recommendation = model2.predict(prediction2_input)[0]

        # Construct the response
        model1_output = f"Initial prediction: {predicted_crop}"
        model2_recommendation_text = f"Tanaman yang disarankan: {model2_recommendation}"

        return CropPredictionResponse(model1_output=model1_output, model2_recommendation=model2_recommendation_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
