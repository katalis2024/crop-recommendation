from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware

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
    model_names = ["KNN_model", "DT_model", "RFC_model", "GBC_model", "XGB_model", "SVM_model", "model2_model"]
    model_dir = "C:\\Users\\ACER\\OneDrive - mail.unnes.ac.id\\katalis\\app\\models"

    for name in model_names:
        model_path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.isfile(model_path):
            with open(model_path, "rb") as file:
                models[name] = pickle.load(file)
                print(f"Successfully loaded {name}")
        else:
            print(f"Warning: {name}.pkl not found in {model_dir}")
    return models

# Load models
models = load_models()

# Accuracies for each model
model_accuracies = {
    "KNN_model": 0.85,
    "DT_model": 0.78,
    "RFC_model": 0.90,
    "GBC_model": 0.88,
    "XGB_model": 0.91,
    "SVM_model": 0.86
}

# Ensemble voting function
def ensemble_voting(predictions_dict, accuracies_dict):
    prediction_counter = Counter(predictions_dict.values())
    if prediction_counter.most_common()[0][1] > len(predictions_dict) / 2:
        return prediction_counter.most_common()[0][0]
    else:
        return max(predictions_dict.items(), key=lambda x: accuracies_dict[x[0]])[1]

# Get recommendations based on parameters
def get_recommendation(N, P, K, Temperature, Humidity, ph, Rainfall):
    average_values = {
        'N': 70, 'P': 45, 'K': 70, 'Temperature': 25,
        'Humidity': 70, 'ph': 6.2, 'Rainfall': 200
    }
    recommendations = []
    for param, pred_value in zip(average_values.keys(), [N, P, K, Temperature, Humidity, ph, Rainfall]):
        avg_value = average_values[param]
        if pred_value < avg_value:
            recommendations.append(f"{param} is deficient (prediction: {pred_value:.2f}). Increase levels.")
        elif pred_value > avg_value:
            recommendations.append(f"{param} is in excess (prediction: {pred_value:.2f}). Reduce levels.")
        else:
            recommendations.append(f"{param} is adequate (prediction: {pred_value:.2f}).")
    return "\n".join(recommendations)

# GET endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API!"}

# POST endpoint for prediction
@app.post("/predict", response_model=ModelOutput)
async def predict_crop(input_data: CropInput):
    try:
        input_features = np.array([[
            input_data.N, input_data.P, input_data.K,
            input_data.Temperature, input_data.Humidity,
            input_data.ph, input_data.Rainfall
        ]])

        # Predictions from Model 1
        predictions_dict = {
            name: model.predict(input_features)[0] 
            for name, model in models.items() 
            if name != "model2_model" and name in model_accuracies
        }

        if not predictions_dict:
            raise ValueError("No valid predictions from Model 1")

        final_prediction = ensemble_voting(predictions_dict, model_accuracies)

        # Prediction with Model 2
        model2 = models.get("model2_model")
        if model2:
            model2_input = np.array([[final_prediction]]).reshape(1, -1)
            model2_recommendation = model2.predict(model2_input)[0]
        else:
            model2_recommendation = "Model 2 not available."

        recommendation_text = get_recommendation(
            input_data.N, input_data.P, input_data.K, input_data.Temperature, 
            input_data.Humidity, input_data.ph, input_data.Rainfall
        )

        return ModelOutput(
            model1_output=str(final_prediction),
            model2_recommendation=f"{model2_recommendation}\n\n{recommendation_text}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)