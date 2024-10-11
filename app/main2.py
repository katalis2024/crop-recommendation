from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import traceback
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    ph: float

class ModelOutput(BaseModel):
    model1_output: str
    model2_recommendation: str
    recommendations: str

def load_models():
    models = {}
    model_names = ["Stacked_model", "model2_model", "scaler", "label_encoder"]
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

models = load_models()

def z_score(value, mean, std):
    return (value - mean) / std

def color_gradient(z):
    norm = mcolors.Normalize(vmin=-3, vmax=3)
    cmap = plt.cm.Blues_r
    rgba = cmap(norm(z))
    hex_color = mcolors.to_hex(rgba)
    return hex_color

def get_recommendation(N, P, K, Temperature, Humidity, ph, scaler, model, label_encoder, mean_values, std_values):
    feature_cols = ["N", "P", "K", "Temperature", "Humidity", "ph"]
    input_data = pd.DataFrame([[N, P, K, Temperature, Humidity, ph]], columns=feature_cols)

    input_data_scaled = scaler.transform(input_data)

    try:
        prediction = model.predict(input_data_scaled)
        predicted_label = label_encoder.inverse_transform([int(prediction[0])])[0]
    except ValueError as e:
        return f"Prediction error: {str(e)}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

    recommendation = []
    for parameter, value in zip(feature_cols, [N, P, K, Temperature, Humidity, ph]):
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API!"}

@app.post("/predict", response_model=ModelOutput)
async def predict_crop(input_data: CropInput):
    try:
        input_features = np.array([[input_data.N, input_data.P, input_data.K,
                                    input_data.Temperature, input_data.Humidity,
                                    input_data.ph]])

        # Model 1 Prediction
        if "Stacked_model" not in models:
            raise ValueError("Model 1 tidak tersedia.")

        model1_output = models["Stacked_model"].predict(input_features)[0]

        # Model 2 Prediction
        if "model2_model" not in models:
            raise ValueError("Model 2 tidak tersedia.")

        model2_input = np.array([[model1_output]])
        model2_recommendation = models["model2_model"].predict(model2_input)[0]

        # Calculate recommendations
        mean_values = {
            'N': 50.55, 'P': 53.36, 'K': 48.15, 'Temperature': 25.62,
            'Humidity': 71.48, 'ph': 6.47
        }

        std_values = {
            'N': 5.0, 'P': 5.0, 'K': 5.0, 'Temperature': 5.0,
            'Humidity': 5.0, 'ph': 1.0
        }

        recommendations = get_recommendation(input_data.N, input_data.P, input_data.K,
                                             input_data.Temperature, input_data.Humidity,
                                             input_data.ph, models["scaler"], 
                                             models["Stacked_model"], models["label_encoder"],
                                             mean_values, std_values)

        return ModelOutput(model1_output=str(model1_output),
                           model2_recommendation=str(model2_recommendation),
                           recommendations=recommendations)

    except Exception as e:
        traceback.print_exc()  # Print traceback for debugging
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
