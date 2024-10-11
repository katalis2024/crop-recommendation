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

# Kelas input untuk data yang akan diprediksi
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    ph: float
    Rainfall: float

# Kelas output untuk hasil prediksi
class ModelOutput(BaseModel):
    model1_output: str
    model2_recommendation: str

# Memuat model-model yang sudah dilatih
def load_models():
    models = {}
    model_names = ["KNN", "DT", "RFC", "GBC", "XGB", "SVM", "model2"]
    for name in model_names:
        model_path = os.path.join("C:\\Users\\ACER\\OneDrive - mail.unnes.ac.id\\katalis\\app", f"{name}_model.pkl")
        print(f"Trying to load model from {model_path}")  # Debug: Print model path
        if os.path.isfile(model_path):
            with open(model_path, "rb") as file:
                models[name] = pickle.load(file)
        else:
            print(f"{model_path} not found!")  # Debug: Print if model not found
    return models

# Load models
models = load_models()
print("Loaded models:", models)  # Debug: Print loaded models

# Akurasi dari masing-masing model
model_accuracies = {
    "KNN": 0.85,
    "DT": 0.78,
    "RFC": 0.90,
    "GBC": 0.88,
    "XGB": 0.91,
    "SVM": 0.86
}

# Fungsi untuk ensemble voting
def ensemble_voting(predictions_dict, accuracies_dict):
    current_predictions = [pred for pred in predictions_dict.values()]
    prediction_counter = Counter(current_predictions)

    # Jika ada mayoritas jelas
    if prediction_counter.most_common()[0][1] > len(predictions_dict) / 2:
        return prediction_counter.most_common()[0][0]
    else:
        best_prediction = max(prediction_counter.most_common(), key=lambda x: accuracies_dict[x[0]])
        return best_prediction[0]

# Fungsi untuk mendapatkan rekomendasi
def get_recommendation(N, P, K, Temperature, Humidity, ph, Rainfall):
    average_values = {
        'N': 70,
        'P': 45,
        'K': 70,
        'Temperature': 25,
        'Humidity': 70,
        'ph': 6.2,
        'Rainfall': 200
    }

    recommendation = []

    for param, pred_value in zip(['N', 'P', 'K', 'Temperature', 'Humidity', 'ph', 'Rainfall'],
                                 [N, P, K, Temperature, Humidity, ph, Rainfall]):
        avg_value = average_values[param]
        if pred_value < avg_value:
            recommendation.append(f"{param} deficient (prediction: {pred_value:.2f}), needs to be increased.")
        elif pred_value > avg_value:
            recommendation.append(f"{param} excess (prediction: {pred_value:.2f}), needs to be reduced.")
        else:
            recommendation.append(f"{param} adequate (prediction: {pred_value:.2f}).")

    return "\n".join(recommendation)

# Endpoint GET
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API!"}

# Endpoint POST untuk memprediksi
@app.post("/predict", response_model=ModelOutput)
async def predict_crop(input_data: CropInput):
    try:
        # Ambil data dari input
        input_features = np.array([[input_data.N, input_data.P, input_data.K,
                                    input_data.Temperature, input_data.Humidity,
                                    input_data.ph, input_data.Rainfall]])

        # Prediksi dengan masing-masing model di Model 1
        predictions_dict = {
            "KNN": models["KNN"].predict(input_features)[0],
            "DT": models["DT"].predict(input_features)[0],
            "RFC": models["RFC"].predict(input_features)[0],
            "GBC": models["GBC"].predict(input_features)[0],
            "XGB": models["XGB"].predict(input_features)[0],
            "SVM": models["SVM"].predict(input_features)[0]
        }

        print("Predictions from models:", predictions_dict)  # Debug: Print model predictions

        # Melakukan ensemble voting
        final_prediction = ensemble_voting(predictions_dict, model_accuracies)

        # Menggunakan hasil dari Model 1 sebagai input untuk Model 2
        prediction_for_model2 = np.array([[final_prediction]])  # Ensure correct shape for Model 2

        # Prediksi dengan Model 2
        model2 = models["model2"]
        model2_recommendation = model2.predict(prediction_for_model2)[0]

        # Mendapatkan rekomendasi berdasarkan output
        recommendation_text = get_recommendation(input_data.N, input_data.P,
                                                 input_data.K, input_data.Temperature, 
                                                 input_data.Humidity, input_data.ph, 
                                                 input_data.Rainfall)

        # Debug: Print final outputs
        print(f"Final prediction from Model 1: {final_prediction}")
        print(f"Recommendation from Model 2: {model2_recommendation}")

        # Return the updated model output structure
        return ModelOutput(
            model1_output=str(final_prediction), 
            model2_recommendation=model2_recommendation
        )

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error message for debugging
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Menjalankan aplikasi
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
