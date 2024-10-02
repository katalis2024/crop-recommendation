from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class for input data
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    ph: float
    Rainfall: float

# Class for model output
class ModelOutput(BaseModel):
    model1_output: str
    model2_recommendation: str

# Load models
def load_models():
    models = {}
    model_names = ["Stacked_model", "model1_model"]
    model_dir = "C:\\Users\\ACER\\OneDrive - mail.unnes.ac.id\\katalis\\app\\models"

    for name in model_names:
        model_path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.isfile(model_path):
            with open(model_path, "rb") as file:
                loaded_model = pickle.load(file)
                models[name] = loaded_model
                print(f"Successfully loaded {name} of type {type(models[name])}")
        else:
            print(f"Warning: {name}.pkl not found in {model_dir}")
    return models

# Load models
models = load_models()

# Get recommendations based on parameters
def get_recommendation(N, P, K, Temperature, Humidity, ph, Rainfall):
    average_values = {
        'N': 50.55, 'P': 53.36, 'K': 48.15, 'Temperature': 25.62,
        'Humidity': 71.48, 'ph': 6.47, 'Rainfall': 103.46
    }
    
    recommendations = []
    color_thresholds = {
        'Red': 15, 'Orange': 5, 'Yellow': 1
    }

    for param, pred_value in zip(average_values.keys(), [N, P, K, Temperature, Humidity, ph, Rainfall]):
        avg_value = average_values[param]
        deviation = pred_value - avg_value
        
        if abs(deviation) > color_thresholds['Red']:
            color = 'Red'
        elif abs(deviation) > color_thresholds['Orange']:
            color = 'Orange'
        else:
            color = 'Yellow'
        
        status = "sedikit berbeda" if color == 'Orange' else "terlalu sedikit" if deviation < 0 else "terlalu tinggi"
        recommendations.append(f"kadar {param} {status} {abs(deviation):.2f} poin dari yang seharusnya {avg_value:.2f}, Warna: {color}")

    return "\n".join(recommendations)

# GET endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API!"}

# POST endpoint for prediction
@app.post("/predict", response_model=ModelOutput)
async def predict_crop(input_data: CropInput):
    try:
        # Prepare input features for model 1
        input_features = np.array([[input_data.N, input_data.P, input_data.K,
                                    input_data.Temperature, input_data.Humidity,
                                    input_data.ph, input_data.Rainfall]])

        # Direct prediction from Model 1
        model1_output = "Model 1 tidak tersedia"
        if "Stacked_model" in models:
            model1_output = models["Stacked_model"].predict(input_features)[0]

        # For Model 2 prediction (if it exists)
        model2_recommendation = "Model 2 tidak tersedia"
        if "model2_model" in models:
            model2_input = np.array([[model1_output]])  # Adjust input shape if necessary
            model2_recommendation = models["model1_model"].predict(model2_input)[0]

        # Get recommendations based on input parameters
        recommendations = get_recommendation(input_data.N, input_data.P, input_data.K,
                                             input_data.Temperature, input_data.Humidity,
                                             input_data.ph, input_data.Rainfall)

        return ModelOutput(model1_output=model1_output, model2_recommendation=model2_recommendation)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
