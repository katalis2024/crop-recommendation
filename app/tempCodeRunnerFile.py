from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import traceback
import pandas as pd  # Pastikan pandas diimpor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt  # Mengimpor matplotlib untuk color gradient
from matplotlib import colors as mcolors

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
    recommendations: str  # Added recommendations output

def load_models():
    models = {}
    model_names = ["Stacked_model", "model2_model"]
    model_dir = "C:\\Users\\ACER\\OneDrive - mail.unnes.ac.id\\katalis\\app\\modelfix"

    for name in model_names:
        model_path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.isfile(model_path):
            with open(model_path, "rb") as file:
                loaded_model = pickle.load(file)
                models[name] = loaded_model
                print(f"Successfully loaded {name} of type {type(models[name])}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    return models

# Load models
models = load_models()

def z_score(value, mean, std):
    """Calculate Z-score for the given value."""
    return (value - mean) / std

def color_gradient(z):
    """Return a color from light to dark based on Z-score."""
    norm = mcolors.Normalize(vmin=-3, vmax=3)  # Normalize Z-scores between -3 and 3
    cmap = plt.cm.Blues_r  # Blue gradient from light to dark
    rgba = cmap(norm(z))  # Get RGBA value based on Z-score
    hex_color = mcolors.to_hex(rgba)  # Convert RGBA to HEX color
    return hex_color

def get_recommendation(N, P, K, Temperature, Humidity, ph, Rainfall, scaler, model, label_encoder, mean_values, std_values):
    """Generate recommendations based on input values and model predictions."""
    feature_cols = ["N", "P", "K", "Temperature", "Humidity", "ph", "Rainfall"]
    input_data = pd.DataFrame([[N, P, K, Temperature, Humidity, ph, Rainfall]], columns=feature_cols)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the crop label using the model
    try:
        prediction = model.predict(input_data_scaled)
        predicted_label = label_encoder.inverse_transform([int(prediction[0])])[0]
    except ValueError as e:
        return f"Prediction error: {str(e)}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

    # Generate the recommendation with color coding
    recommendation = []
    for parameter, value in zip(feature_cols, [N, P, K, Temperature, Humidity, ph, Rainfall]):
        mean_value = mean_values[parameter]
        std_value = std_values[parameter]
        z = z_score(value, mean_value, std_value)
        
        if value < mean_value:
            adjustment = mean_value - value
            color = color_gradient(z)
            recommendation.append(f"<span style='color:{color}'>Sarankan menambahkan {adjustment:.2f} untuk {parameter}.</span>")
        elif value > mean_value:
            adjustment = value - mean_value
            color = color_gradient(z)
            recommendation.append(f"<span style='color:{color}'>Sarankan mengurangi {adjustment:.2f} untuk {parameter}.</span>")
    
    recommendation.append(f"Tanaman yang disarankan: {predicted_label}")

    return "\n".join(recommendation)

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

        # Direct prediction from Stacked Model (Model 1)
        model1_output = None
        if "Stacked_model" in models:
            model1_output = models["Stacked_model"].predict(input_features)[0]

        # Ensure model1_output is a string
        if model1_output is None:
            model1_output = "Model 1 tidak tersedia"

        # For Model 2 prediction (assuming it takes output from Model 1)
        model2_recommendation = None
        if "model1_model" in models:
            model2_input = np.array([[model1_output]])  # Input shape for regression model
            model2_recommendation = models["model2_model"].predict(model2_input)[0]

        # Ensure model2_recommendation is a string
        if model2_recommendation is None:
            model2_recommendation = "Model 2 tidak tersedia"

        # Get mean and std values for recommendations
        mean_values = {
            'N': 50.55, 'P': 53.36, 'K': 48.15, 'Temperature': 25.62,
            'Humidity': 71.48, 'ph': 6.47, 'Rainfall': 103.46
        }
        
        std_values = {
            'N': 5.0, 'P': 5.0, 'K': 5.0, 'Temperature': 5.0,
            'Humidity': 5.0, 'ph': 1.0, 'Rainfall': 10.0
        }
        
        # Get recommendations based on input parameters
        recommendations = get_recommendation(input_data.N, input_data.P, input_data.K,
                                             input_data.Temperature, input_data.Humidity,
                                             input_data.ph, input_data.Rainfall,
                                             models["scaler"], models["model"], models["label_encoder"],
                                             mean_values, std_values)

        return ModelOutput(model1_output=str(model1_output),
                           model2_recommendation=str(model2_recommendation),
                           recommendations=recommendations)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
