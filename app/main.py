from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

# Paths to the datasets
TRAINING_DATASET_PATH = r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app\data\BackupCrop_Recommendation .csv'
PARAMETER_DATASET_PATH = r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app\data\Crop_Recommendation.csv'
OUTPUT_DIR = 'output'


# Load datasets
try:
    parameter_data = pd.read_csv(r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app\data\Crop_Recommendation.csv')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Parameter dataset file not found.")

# Utility function to load models
def load_model(file_path: str):
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    raise FileNotFoundError(f"File not found: {file_path}")

# Load model and scaler files
try:
    scaler = load_model(r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app\scaler_fixmodel.pkl')
    label_encoder = load_model(r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app\label_encoder_fixmodel.pkl')
    model1 = load_model(os.path.join(OUTPUT_DIR, r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app\SStacked_model.pkl'))
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=str(e))


# Data models
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    ph: float

class ParameterDetail(BaseModel):
    class_: str
    colour: str
    value: float
    deviation: float
    satuan: str
    action: str

class CropPredictionResponse(BaseModel):
    model1_output: str
    model2_recommendation: str
    N: ParameterDetail
    P: ParameterDetail
    K: ParameterDetail
    Temperature: ParameterDetail
    Humidity: ParameterDetail
    ph: ParameterDetail

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Utility for parameter adequacy
def categorize_parameter(value: float, mean: float, min_val: float, max_val: float) -> tuple:
    deviation = (value - mean) / mean
    if deviation > 0.1:
        return "excess", "red", deviation
    elif deviation < -0.1:
        return "deficiency", "orange", deviation
    return "adequate", "green", deviation

@app.post("/predict/", response_model=CropPredictionResponse)
def predict_crop(input_data: CropInput):
    try:
        # Scale and predict
        input_features = np.array([[input_data.N, input_data.P, input_data.K,
                                    input_data.Temperature, input_data.Humidity, input_data.ph]])
        input_scaled = scaler.transform(input_features)
        prediction1 = model1.predict(input_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction1)[0]

        # Retrieve parameters for predicted crop
        crop_row = parameter_data[parameter_data['label'] == predicted_crop]
        if crop_row.empty:
            raise HTTPException(status_code=404, detail=f"No data for crop '{predicted_crop}' in dataset.")
        
        # Fetching the values
        try:
            N_mean, N_min, N_max = crop_row[['N_mean', 'N_min', 'N_max']].values[0]
            P_mean, P_min, P_max = crop_row[['P_mean', 'P_min', 'P_max']].values[0]
            K_mean, K_min, K_max = crop_row[['K_mean', 'K_min', 'K_max']].values[0]
            Temperature_mean, Temperature_min, Temperature_max = crop_row[['temperature_mean', 'temperature_min', 'temperature_max']].values[0]
            Humidity_mean, Humidity_min, Humidity_max = crop_row[['Humidity_mean', 'Humidity_min', 'Humidity_max']].values[0]
            ph_mean, ph_min, ph_max = crop_row[['ph_mean', 'ph_min', 'ph_max']].values[0]
        except KeyError as e:
            raise HTTPException(status_code=500, detail=f"Missing column: {str(e)}")

        # Categorize parameters
        N_class, N_colour, N_deviation = categorize_parameter(input_data.N, N_mean, N_min, N_max)
        P_class, P_colour, P_deviation = categorize_parameter(input_data.P, P_mean, P_min, P_max)
        K_class, K_colour, K_deviation = categorize_parameter(input_data.K, K_mean, K_min, K_max)
        Temperature_class, Temperature_colour, Temperature_deviation = categorize_parameter(input_data.Temperature, Temperature_mean, Temperature_min, Temperature_max)
        Humidity_class, Humidity_colour, Humidity_deviation = categorize_parameter(input_data.Humidity, Humidity_mean, Humidity_min, Humidity_max)
        ph_class, ph_colour, ph_deviation = categorize_parameter(input_data.ph, ph_mean, ph_min, ph_max)

        # Response assembly
        return CropPredictionResponse(
            model1_output=predicted_crop,
            model2_recommendation=f"Recommended crop: {predicted_crop}",
            N=ParameterDetail(class_=N_class, colour=N_colour, value=input_data.N, deviation=N_deviation, satuan="mg", action="adjust" if N_class != "adequate" else "maintain"),
            P=ParameterDetail(class_=P_class, colour=P_colour, value=input_data.P, deviation=P_deviation, satuan="mg", action="adjust" if P_class != "adequate" else "maintain"),
            K=ParameterDetail(class_=K_class, colour=K_colour, value=input_data.K, deviation=K_deviation, satuan="mg", action="adjust" if K_class != "adequate" else "maintain"),
            Temperature=ParameterDetail(class_=Temperature_class, colour=Temperature_colour, value=input_data.Temperature, deviation=Temperature_deviation, satuan="°C", action="adjust" if Temperature_class != "adequate" else "maintain"),
            Humidity=ParameterDetail(class_=Humidity_class, colour=Humidity_colour, value=input_data.Humidity, deviation=Humidity_deviation, satuan="%", action="adjust" if Humidity_class != "adequate" else "maintain"),
            ph=ParameterDetail(class_=ph_class, colour=ph_colour, value=input_data.ph, deviation=ph_deviation, satuan="pH", action="adjust" if ph_class != "adequate" else "maintain")
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
