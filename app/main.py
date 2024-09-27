from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from collections import Counter

app = FastAPI()

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
    predicted_label: str
    recommendations: str

# Memuat model-model yang sudah dilatih
with open("KNN_model.pkl", "rb") as file:
    knn_model = pickle.load(file)

with open("DT_model.pkl", "rb") as file:
    dt_model = pickle.load(file)

with open("RFC_model.pkl", "rb") as file:
    rfc_model = pickle.load(file)

with open("GBC_model.pkl", "rb") as file:
    gbc_model = pickle.load(file)

with open("XGB_model.pkl", "rb") as file:
    xgb_model = pickle.load(file)

with open("SVM_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("model2.pkl", "rb") as file:
    model2 = pickle.load(file)

# Akurasi dari masing-masing model (disesuaikan dengan nilai sebenarnya dari model Anda)
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
    final_predictions = []
    for i in range(len(next(iter(predictions_dict.values())))):  # Panjang dari data uji
        current_predictions = [pred[i] for pred in predictions_dict.values()]
        prediction_counter = Counter(current_predictions)
        most_common_prediction, count = prediction_counter.most_common(1)[0]

        if count > 4:
            final_predictions.append(most_common_prediction)
        elif count == 3 and len(prediction_counter) == 3:
            top_two_predictions = prediction_counter.most_common(2)
            best_prediction = max(top_two_predictions, key=lambda x: accuracies_dict[x[0]])
            final_predictions.append(best_prediction[0])
        elif count == 2 and len(prediction_counter) == 4:
            top_three_predictions = prediction_counter.most_common(3)
            best_prediction = max(top_three_predictions, key=lambda x: accuracies_dict[x[0]])
            final_predictions.append(best_prediction[0])
        else:
            final_predictions.append(most_common_prediction)

    return final_predictions

# Fungsi untuk mendapatkan rekomendasi
def get_recommendation(label, N, P, K, Temperature, Humidity, ph, Rainfall):
    plant_ranges = {
        'rice': {
            'N': (50, 100),
            'P': (30, 60),
            'K': (50, 90),
            'Temperature': (20, 30),
            'Humidity': (50, 90),
            'ph': (5.5, 7.0),
            'Rainfall': (150, 300)
        },
        'maize': {
            'N': (40, 90),
            'P': (20, 50),
            'K': (40, 80),
            'Temperature': (18, 35),
            'Humidity': (40, 80),
            'ph': (5.5, 6.8),
            'Rainfall': (100, 250)
        },
        # Tambahkan lebih banyak tanaman di sini...
    }

    if label not in plant_ranges:
        return f"Tanaman '{label}' tidak ditemukan dalam database."

    ranges = plant_ranges[label]
    recommendation = []

    for param, pred_value in zip(['N', 'P', 'K', 'Temperature', 'Humidity', 'ph', 'Rainfall'],
                                 [N, P, K, Temperature, Humidity, ph, Rainfall]):
        min_val, max_val = ranges[param]
        if pred_value < min_val:
            recommendation.append(f"{param} kekurangan (prediksi: {pred_value:.2f}), perlu ditingkatkan.")
        elif pred_value > max_val:
            recommendation.append(f"{param} berlebih (prediksi: {pred_value:.2f}), perlu dikurangi.")
        else:
            recommendation.append(f"{param} sesuai (prediksi: {pred_value:.2f}).")

    return "\n".join(recommendation)

# Endpoint GET
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API!"}

# Endpoint POST untuk memprediksi
@app.post("/predict", response_model=ModelOutput)
async def predict_crop(input_data: CropInput):
    # Mengambil data dari input
    input_features = np.array([[input_data.N, input_data.P, input_data.K,
                                input_data.Temperature, input_data.Humidity,
                                input_data.ph, input_data.Rainfall]])

    # Prediksi dengan masing-masing model di Model 1
    predictions_dict = {
        "KNN": knn_model.predict(input_features),
        "DT": dt_model.predict(input_features),
        "RFC": rfc_model.predict(input_features),
        "GBC": gbc_model.predict(input_features),
        "XGB": xgb_model.predict(input_features),
        "SVM": svm_model.predict(input_features)
    }

    # Melakukan ensemble voting
    final_prediction = ensemble_voting(predictions_dict, model_accuracies)[0]

    # Menggunakan hasil dari Model 1 sebagai input untuk Model 2
    prediction_for_model2 = input_features

    # Prediksi dengan Model 2
    recommendations = model2.predict(prediction_for_model2)

    # Mendapatkan rekomendasi berdasarkan output
    recommendation_text = get_recommendation(final_prediction, input_data.N, input_data.P,
                                             input_data.K, input_data.Temperature, 
                                             input_data.Humidity, input_data.ph, 
                                             input_data.Rainfall)

    return ModelOutput(predicted_label=final_prediction, recommendations=recommendation_text)

# Menjalankan aplikasi
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
