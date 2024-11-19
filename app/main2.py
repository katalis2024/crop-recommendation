from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI()

# Load model function
def load_model(file_path: str):
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def is_docker():
    return os.path.exists('/.dockerenv')

# Tentukan path yang sesuai berdasarkan lingkungan
if is_docker():
    OUTPUT_DIR = '/app'  # Path di dalam Docker container
else:
    OUTPUT_DIR = r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app'  # Path di lingkungan lokal

# Tentukan path file model berdasarkan direktori yang dipilih
scaler_path = os.path.join(OUTPUT_DIR, 'pscaler_fixmodel.pkl')
label_encoder_path = os.path.join(OUTPUT_DIR, 'plabel_encoder_fixmodel.pkl')
model1_path = os.path.join(OUTPUT_DIR, 'PStacked_model.pkl')

# Cek jika file ada
if os.path.exists(scaler_path) and os.path.exists(label_encoder_path) and os.path.exists(model1_path):
    print("Model files found!")
else:
    print("Model files not found.")
# Load models and scalers
try:
    label_encoder = load_model(label_encoder_path)
    model1 = load_model(model1_path)
    scaler = load_model(scaler_path)
    logger.info("Models and scaler loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Error loading models or scaler: {e}")
    raise RuntimeError(f"Failed to initialize models: {e}")

# Parameter data (example data)

parameter_data =   {
            "apple": {"N_mean": 20.8, "N_min": 0, "N_max": 40, "P_mean": 134.22, "P_min": 120, "P_max": 145,
                    "K_mean": 199.89, "K_min": 120, "K_max": 200, "Humidity_mean": 92.33, "Humidity_min": 90.03, "Humidity_max": 94.92,
                    "ph_mean": 5.93, "ph_min": 5.0, "ph_max": 7.5, "temperature_mean": 24.09, "temperature_min": 15.0, "temperature_max": 35.0},
            "banana": {"N_mean": 100.23, "N_min": 80, "N_max": 120, "P_mean": 82.01, "P_min": 70, "P_max": 95,
                    "K_mean": 50.05, "K_min": 0, "K_max": 120, "Humidity_mean": 80.36, "Humidity_min": 75.03, "Humidity_max": 84.98,
                    "ph_mean": 5.98, "ph_min": 5.0, "ph_max": 7.5, "temperature_mean": 26.12, "temperature_min": 20.0, "temperature_max": 35.0},
            "blackgram": {"N_mean": 40.02, "N_min": 20, "N_max": 60, "P_mean": 67.47, "P_min": 55, "P_max": 80,
                        "K_mean": 19.24, "K_min": 0, "K_max": 80, "Humidity_mean": 65.12, "Humidity_min": 60.07, "Humidity_max": 69.96,
                        "ph_mean": 7.13, "ph_min": 6.0, "ph_max": 8.0, "temperature_mean": 29.91, "temperature_min": 22.0, "temperature_max": 35.0},
            "chickpea": {"N_mean": 40.09, "N_min": 20, "N_max": 60, "P_mean": 67.79, "P_min": 55, "P_max": 80,
                        "K_mean": 79.92, "K_min": 55, "K_max": 80, "Humidity_mean": 16.86, "Humidity_min": 14.26, "Humidity_max": 19.97,
                        "ph_mean": 7.34, "ph_min": 6.5, "ph_max": 8.0, "temperature_mean": 21.11, "temperature_min": 15.0, "temperature_max": 30.0},
            "coconut": {"N_mean": 21.98, "N_min": 0, "N_max": 40, "P_mean": 16.93, "P_min": 5, "P_max": 30,
                        "K_mean": 30.59, "K_min": 5, "K_max": 30, "Humidity_mean": 94.84, "Humidity_min": 90.02, "Humidity_max": 99.98,
                        "ph_mean": 5.98, "ph_min": 5.0, "ph_max": 7.0, "temperature_mean": 28.2, "temperature_min": 20.0, "temperature_max": 35.0},
            "coffee": {"N_mean": 101.2, "N_min": 80, "N_max": 120, "P_mean": 28.74, "P_min": 15, "P_max": 40,
                    "K_mean": 29.94, "K_min": 15, "K_max": 40, "Humidity_mean": 58.87, "Humidity_min": 50.05, "Humidity_max": 69.95,
                    "ph_mean": 6.79, "ph_min": 6.0, "ph_max": 7.0, "temperature_mean": 21.55, "temperature_min": 15.0, "temperature_max": 30.0},
            "cotton": {"N_mean": 117.77, "N_min": 100, "N_max": 140, "P_mean": 46.24, "P_min": 35, "P_max": 60,
                    "K_mean": 19.56, "K_min": 19, "K_max": 60, "Humidity_mean": 79.84, "Humidity_min": 75.01, "Humidity_max": 84.88,
                    "ph_mean": 6.91, "ph_min": 6.0, "ph_max": 8.0, "temperature_mean": 28.01, "temperature_min": 20.0, "temperature_max": 35.0},
            "grapes": {"N_mean": 23.18, "N_min": 0, "N_max": 40, "P_mean": 132.53, "P_min": 120, "P_max": 145,
                    "K_mean": 200.11, "K_min": 120, "K_max": 200, "Humidity_mean": 81.87, "Humidity_min": 80.02, "Humidity_max": 83.98,
                    "ph_mean": 6.03, "ph_min": 5.5, "ph_max": 7.0, "temperature_mean": 24.35, "temperature_min": 15.0, "temperature_max": 35.0},
            "jute": {"N_mean": 78.4, "N_min": 60, "N_max": 100, "P_mean": 46.86, "P_min": 35, "P_max": 60,
                    "K_mean": 39.99, "K_min": 15, "K_max": 60, "Humidity_mean": 79.64, "Humidity_min": 70.88, "Humidity_max": 89.89,
                    "ph_mean": 6.73, "ph_min": 6.5, "ph_max": 7.5, "temperature_mean": 26.87, "temperature_min": 20.0, "temperature_max": 35.0},
            "kidneybeans": {"N_mean": 20.75, "N_min": 0, "N_max": 40, "P_mean": 67.54, "P_min": 55, "P_max": 80,
                            "K_mean": 20.05, "K_min": 19, "K_max": 80, "Humidity_mean": 21.61, "Humidity_min": 18.09, "Humidity_max": 24.97,
                            "ph_mean": 5.75, "ph_min": 5.0, "ph_max": 7.5, "temperature_mean": 25.89, "temperature_min": 15.0, "temperature_max": 30.0},
            "lentil": {"N_mean": 18.77, "N_min": 0, "N_max": 40, "P_mean": 68.36, "P_min": 55, "P_max": 80,
                    "K_mean": 19.41, "K_min": 19, "K_max": 80, "Humidity_mean": 64.8, "Humidity_min": 60.09, "Humidity_max": 69.92,
                    "ph_mean": 6.93, "ph_min": 6.0, "ph_max": 8.0, "temperature_mean": 30.17, "temperature_min": 20.0, "temperature_max": 35.0},
            "maize": {"N_mean": 77.76, "N_min": 60, "N_max": 100, "P_mean": 48.44, "P_min": 35, "P_max": 60,
                    "K_mean": 19.79, "K_min": 15, "K_max": 60, "Humidity_mean": 65.09, "Humidity_min": 55.28, "Humidity_max": 74.83,
                    "ph_mean": 6.25, "ph_min": 5.5, "ph_max": 7.5, "temperature_mean": 27.54, "temperature_min": 22.0, "temperature_max": 35.0},
            "mango": {"N_mean": 20.07, "N_min": 0, "N_max": 40, "P_mean": 27.18, "P_min": 15, "P_max": 40,
                    "K_mean": 29.92, "K_min": 15, "K_max": 40, "Humidity_mean": 50.16, "Humidity_min": 45.02, "Humidity_max": 54.96,
                    "ph_mean": 5.77, "ph_min": 5.0, "ph_max": 7.0, "temperature_mean": 25.56, "temperature_min": 20.0, "temperature_max": 35.0},
            "mothbeans":    {"N_mean": 21.44, "N_min": 0, "N_max": 40, "P_mean": 48.01, "P_min": 35, "P_max": 60,
                            "K_mean": 20.23, "K_min": 19, "K_max": 60, "Humidity_mean": 53.16, "Humidity_min": 40.01, "Humidity_max": 64.96,
                            "ph_mean": 6.83, "ph_min": 6.0, "ph_max": 7.0, "temperature_mean": 26.12, "temperature_min": 20.0, "temperature_max": 35.0},
            "mungbean": {    "N_mean": 20.99, "N_min": 0, "N_max": 40, "P_mean": 47.28, "P_min": 35, "P_max": 60,
                            "K_mean": 19.87, "K_min": 19, "K_max": 60,"Humidity_mean": 85.5, "Humidity_min": 80.03, "Humidity_max": 89.99,
                            "ph_mean": 6.72, "ph_min": 5.0, "ph_max": 7.0,"temperature_mean": 25.32, "temperature_min": 20.0, "temperature_max": 35.0},
            "muskmelon": {  "N_mean": 100.32, "N_min": 80, "N_max": 120, "P_mean": 17.72, "P_min": 5, "P_max": 30,
                            "K_mean": 50.08, "K_min": 5, "K_max": 30,"Humidity_mean": 92.34, "Humidity_min": 90.02, "Humidity_max": 94.96,
                            "ph_mean": 6.36, "ph_min": 5.0, "ph_max": 7.0,"temperature_mean": 25.43, "temperature_min": 20.0, "temperature_max": 35.0 },
            "orange": {
                "N_mean": 19.58, "N_min": 0, "N_max": 40, 
                "P_mean": 16.55, "P_min": 5, "P_max": 30,
                "K_mean": 10.01, "K_min": 5, "K_max": 30,
                "Humidity_mean": 92.17, "Humidity_min": 90.01, "Humidity_max": 94.96,
                "ph_mean": 7.02, "ph_min": 5.0, "ph_max": 7.5,
                "temperature_mean": 25.02, "temperature_min": 20.0, "temperature_max": 35.0
            },
            "papaya": {
                "N_mean": 49.88, "N_min": 31, "N_max": 70, 
                "P_mean": 59.05, "P_min": 46, "P_max": 70,
                "K_mean": 50.04, "K_min": 46, "K_max": 70,
                "Humidity_mean": 92.4, "Humidity_min": 90.04, "Humidity_max": 94.94,
                "ph_mean": 6.74, "ph_min": 5.0, "ph_max": 7.0,
                "temperature_mean": 24.87, "temperature_min": 20.0, "temperature_max": 35.0
            },
            "pigeonpeas": {
                "N_mean": 20.73, "N_min": 0, "N_max": 40, 
                "P_mean": 67.73, "P_min": 55, "P_max": 80,
                "K_mean": 20.29, "K_min": 55, "K_max": 80,
                "Humidity_mean": 48.06, "Humidity_min": 30.4, "Humidity_max": 69.69,
                "ph_mean": 5.79, "ph_min": 5.0, "ph_max": 7.0,
                "temperature_mean": 27.45, "temperature_min": 20.0, "temperature_max": 35.0
            },
            "pomegranate": {
                "N_mean": 18.87, "N_min": 0, "N_max": 40, 
                "P_mean": 18.75, "P_min": 5, "P_max": 30,
                "K_mean": 40.21, "K_min": 5, "K_max": 30,
                "Humidity_mean": 90.13, "Humidity_min": 85.13, "Humidity_max": 94.99,
                "ph_mean": 6.43, "ph_min": 5.0, "ph_max": 7.0,
                "temperature_mean": 28.76, "temperature_min": 20.0, "temperature_max": 35.0
            },
            "rice": {
                "N_mean": 79.89, "N_min": 60, "N_max": 99, 
                "P_mean": 47.58, "P_min": 35, "P_max": 60,
                "K_mean": 39.87, "K_min": 35, "K_max": 60,
                "Humidity_mean": 82.27, "Humidity_min": 80.12, "Humidity_max": 84.97,
                "ph_mean": 6.43, "ph_min": 5.5, "ph_max": 7.5,
                "temperature_mean": 26.03, "temperature_min": 20.0, "temperature_max": 35.0
            },
            "watermelon": {
                "N_mean": 99.42, "N_min": 80, "N_max": 120, 
                "P_mean": 17.0, "P_min": 5, "P_max": 30,
                "K_mean": 50.22, "K_min": 5, "K_max": 30,
                "Humidity_mean": 85.16, "Humidity_min": 80.03, "Humidity_max": 89.98,
                "ph_mean": 6.5, "ph_min": 5.5, "ph_max": 7.5,
                "temperature_mean": 24.73, "temperature_min": 20.0, "temperature_max": 35.0
            }
        }




# Input data model
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    ph: float

# Helper function to calculate stats
def calculate_parameter_stats(value, param_name, crop_params):
    mean = crop_params.get(f"{param_name}_mean")
    min_val = crop_params.get(f"{param_name}_min")
    max_val = crop_params.get(f"{param_name}_max")
    if value < min_val:
        status = "Deficient"
    elif value > max_val:
        status = "Excess"
    else:
        status = "Optimal"
    return {"value": value, "mean": mean, "status": status}

# Prediction route
@app.post("/predict/")
async def predict_crop(input_data: CropInput):
    try:
        # Prepare input features
        input_features = np.array([input_data.N, input_data.P, input_data.K, 
                                    input_data.Temperature, input_data.Humidity, input_data.ph]).reshape(1, -1)
        logger.debug(f"Input features: {input_features}")

        # Scale input features
        input_scaled = scaler.transform(input_features)
        logger.debug(f"Scaled input features: {input_scaled}")

        # Make prediction
        prediction = model1.predict(input_scaled).ravel()
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        logger.debug(f"Predicted crop: {predicted_crop}")

        # Get crop parameters
        crop_params = parameter_data.get(predicted_crop.lower())
        if not crop_params:
            raise HTTPException(status_code=404, detail=f"No parameter data for crop: {predicted_crop}")

        # Calculate parameter stats
        params = {
            "N": calculate_parameter_stats(input_data.N, "N", crop_params),
            "P": calculate_parameter_stats(input_data.P, "P", crop_params),
            "K": calculate_parameter_stats(input_data.K, "K", crop_params),
            "Temperature": calculate_parameter_stats(input_data.Temperature, "temperature", crop_params),
            "Humidity": calculate_parameter_stats(input_data.Humidity, "Humidity", crop_params),
            "ph": calculate_parameter_stats(input_data.ph, "ph", crop_params)
        }

        # Return response
        return {
            "Crop": predicted_crop,
            "Parameters": params
        }
    except ValueError as ve:
        logger.error(f"Value error: {ve}")
        raise HTTPException(status_code=400, detail=f"Input data error: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
