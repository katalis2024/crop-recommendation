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
    # model2 = load_model(os.path.join(OUTPUT_DIR, 'model2.pkl'))  # Load Model 2 for recommendation
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
class ParameterDetail(BaseModel):
    class_: str
    colour: str
    value: float
    deviation: float
    satuan: str
    action: str

class CropPredictionResponse(BaseModel):
    Crop: str
    N: ParameterDetail
    P: ParameterDetail
    K: ParameterDetail
    Temperature: ParameterDetail
    Humidity: ParameterDetail
    ph: ParameterDetail

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

        
        response_data = CropPredictionResponse(
            Crop=predicted_crop,
            N=ParameterDetail(
                class_="sedikit lebih",
                colour="yellow",
                value=90.00,
                deviation=0.13,
                satuan="mg",
                action="decrease"
            ),
            P=ParameterDetail(
                class_="sedikit kurang",
                colour="yellow",
                value=42.00,
                deviation=0.12,
                satuan="mg",
                action="increase"
            ),
            K=ParameterDetail(
                class_="cukup",
                colour="green",
                value=43.00,
                deviation=0.08,
                satuan="mg",
                action="maintain"
            ),
            Temperature=ParameterDetail(
                class_="sedikit kurang",
                colour="yellow",
                value=20.87,
                deviation=0.12,
                satuan="°C",
                action="increase"
            ),
            Humidity=ParameterDetail(
                class_="cukup",
                colour="green",
                value=82.00,
                deviation=0.00,
                satuan="%",
                action="maintain"
            ),
            ph=ParameterDetail(
                class_="cukup",
                colour="green",
                value=6.50,
                deviation=0.01,
                satuan="pH",
                action="maintain"
            )
        )

        return response_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
