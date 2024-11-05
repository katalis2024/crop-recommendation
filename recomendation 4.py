import os
import sys
import pandas as pd
import pickle
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/BackupCrop_Recommendation .csv'
MODEL_FILE = 'scaler.pkl'
MODEL_FILE = 'label_encoder.pkl'

# Function to load the dataset
def load_dataset(path):
    if not os.path.exists(path):
        print(f"File not found at: {path}")
        sys.exit()
    return pd.read_csv(path)

# Function to preprocess data
def preprocess_data(data):
    features = data[['N', 'P', 'K', 'Temperature', 'Humidity', 'ph']]
    labels = data['label']
    return features, labels

# Function to save models
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Function to load models
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load the dataset
crop_data = load_dataset(DATASET_PATH)

# Preprocess the data
features, labels = preprocess_data(crop_data)

# Create and fit the scaler and label encoder
scaler = StandardScaler().fit(features)
label_encoder = LabelEncoder().fit(labels)

# Save the scaler and label encoder
save_model(scaler, 'scaler.pkl')
save_model(label_encoder, 'label_encoder.pkl')

# Load the scaler and label encoder
scaler = load_model('scaler.pkl')
label_encoder = load_model('label_encoder.pkl')

# Load the stacked model
stacked_model = load_model(os.path.join('output', 'stacked_model.pkl'))

    
# Integrated crop data
data = {
    'label': ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon'],
    'N_mean': [20.80, 100.23, 40.02, 40.09, 21.98, 101.20, 117.77, 23.18, 78.40, 20.75, 18.77, 77.76, 20.07, 21.44, 20.99, 100.32, 19.58, 49.88, 20.73, 18.87, 79.89, 99.42],
    'N_median': [24.0, 100.5, 41.0, 39.0, 24.0, 103.0, 117.0, 24.0, 78.0, 22.0, 16.5, 76.0, 21.0, 22.0, 22.0, 100.0, 19.0, 49.0, 20.0, 18.0, 80.0, 99.0],
    'N_min': [0, 80, 20, 20, 0, 80, 100, 0, 60, 0, 0, 60, 0, 0, 0, 80, 0, 31, 0, 0, 60, 80],
    'N_max': [40, 120, 60, 60, 40, 120, 140, 40, 100, 40, 40, 100, 40, 40, 40, 120, 40, 70, 40, 40, 99, 120],
    'P_mean': [134.22, 82.01, 67.47, 67.79, 16.93, 28.74, 46.24, 132.53, 46.86, 67.54, 68.36, 48.44, 27.18, 48.01, 47.28, 17.72, 16.55, 59.05, 67.73, 18.75, 47.58, 17.00],
    'P_median': [136.5, 81.0, 67.0, 68.0, 15.5, 29.0, 46.0, 133.0, 46.0, 67.0, 68.0, 48.5, 27.5, 48.5, 47.0, 18.0, 16.0, 60.0, 69.5, 20.0, 47.0, 17.5],
    'P_min': [120, 70, 55, 55, 5, 15, 35, 120, 35, 55, 55, 35, 15, 35, 35, 5, 5, 46, 55, 5, 35, 5],
    'P_max': [145, 95, 80, 80, 30, 40, 60, 145, 60, 80, 80, 60, 40, 60, 60, 30, 30, 70, 80, 30, 60, 30],
    'K_mean': [199.89, 50.05, 19.24, 79.92, 30.59, 29.94, 19.56, 200.11, 39.99, 20.05, 19.41, 19.79, 29.92, 20.23, 19.87, 50.08, 10.01, 50.04, 20.29, 40.21, 39.87, 50.22],
    'K_median': [200.0, 50.0, 19.0, 79.0, 31.0, 30.0, 19.0, 201.0, 40.0, 20.0, 19.0, 20.0, 30.0, 20.0, 20.0, 50.0, 10.0, 50.0, 20.0, 40.0, 40.0, 50.5],
    'K_min': [120, 0, 0, 55, 5, 15, 19, 120, 15, 19, 19, 15, 15, 19, 19, 5, 5, 46, 55, 5, 35, 5],
    'K_max': [200, 120, 80, 80, 30, 40, 60, 200, 60, 80, 80, 60, 40, 60, 60, 30, 30, 70, 80, 30, 60, 30],
    'Humidity_mean': [92.33, 80.36, 65.12, 16.86, 94.84, 58.87, 79.84, 81.87, 79.64, 21.61, 64.80, 65.09, 50.16, 53.16, 85.50, 92.34, 92.17, 92.40, 48.06, 90.13, 82.27, 85.16],
    'Humidity_median': [92.42, 80.22, 65.03, 16.66, 94.96, 57.65, 80.01, 81.72, 79.47, 21.35, 64.09, 65.30, 50.28, 53.67, 85.95, 92.11, 91.96, 92.68, 47.20, 89.91, 82.19, 85.03],
    'Humidity_min': [90.03, 75.03, 60.07, 14.26, 90.02, 50.05, 75.01, 80.02, 70.88, 18.09, 60.09, 55.28, 45.02, 40.01, 80.03, 90.02, 90.01, 90.04, 30.40, 85.13, 80.12, 80.03],
    'Humidity_max': [94.92, 84.98, 69.96, 19.97, 99.98, 69.95, 84.88, 83.98, 89.89, 24.97, 69.92, 74.83, 54.96, 64.96, 89.99, 94.96, 94.96, 94.94, 69.69, 94.99, 84.97, 89.98],
    'ph_mean': [5.93, 5.98, 7.13, 7.34, 5.98, 6.79, 6.91, 6.03, 6.73, 5.75, 6.93, 6.25, 5.77, 6.83, 6.72, 6.36, 7.02, 6.74, 5.79, 6.43, 6.43, 6.50],
    'ph_median': [5.89, 5.99, 7.17, 7.36, 5.99, 6.80, 6.84, 6.00, 6.71, 5.75, 6.95, 6.26, 5.74, 7.22, 6.70, 6.35, 7.01, 6.78, 5.80, 6.41, 6.44, 6.51],
    'ph_min': [5.00, 5.00, 6.00, 6.50, 5.00, 6.00, 6.50, 5.00, 5.50, 5.00, 6.00, 5.00, 5.00, 6.50, 6.50, 5.50, 6.50, 6.00, 5.50, 6.00, 6.00, 6.00],
    'ph_max': [7.00, 7.50, 8.00, 8.50, 6.50, 7.00, 7.50, 7.00, 7.50, 7.00, 7.50, 7.00, 6.50, 8.00, 8.00, 7.00, 7.50, 7.50, 6.50, 7.50, 7.50, 7.50],
    'Temperature_mean': [22.630942, 27.376798, 29.973340, 18.872847, 27.409892, 25.540477, 23.988958, 23.849575, 24.958376, 20.115085, 24.509052, 22.389204, 31.208770, 28.194920, 28.525775, 28.663066, 22.765725, 33.723859, 27.741762, 21.837842, 23.689332, 25.591767],
    'Temperature_median': [22.628290, 27.443333, 29.655515, 18.878291, 27.385317, 25.656643, 23.964997, 23.018528, 24.971106, 19.924037, 24.946835, 22.844456, 31.300223, 28.370863, 28.441673, 28.851775, 22.901055, 33.262870, 28.931707, 22.354425, 23.734837, 25.603965],
    'Temperature_min': [21.036527, 25.010185, 25.097374, 17.024985, 25.008724, 23.059519, 22.000851, 8.825675, 23.094338, 15.330426, 18.064861, 18.041855, 27.003155, 24.018254, 27.014704, 27.024151, 10.010813, 23.012402, 18.319104, 18.071330, 20.045414, 24.043558],
    'Temperature_max': [23.996862, 29.908885, 34.946616, 20.995022, 29.869083, 27.923744, 25.992374, 41.948657, 26.985822, 24.923601, 29.944139, 26.549864, 35.990097, 31.999286, 29.914544, 29.943492, 34.906653, 43.675493, 36.977944, 24.962732, 26.929951, 26.986037]
 }


# Convert data to DataFrame
crop_df = pd.DataFrame(data)

def determine_class(value, min_val, max_val, mean_val):
    if value < min_val:
        return "kurang"
    elif value > max_val:
        return "lebih"
    else:
        deviation = abs(value - mean_val) / mean_val
        if deviation <= 0.1:  # Within 10% of mean
            return "cukup"
        elif value < mean_val:
            return "sedikit kurang"
        else:
            return "sedikit lebih"

def determine_color(class_val):
    if class_val == "kurang":
        return "red"
    elif class_val == "lebih":
        return "blue"
    elif class_val == "cukup":
        return "green"
    else:
        return "yellow"

def determine_action(class_val):
    if class_val == "kurang" or class_val == "sedikit kurang":
        return "increase"
    elif class_val == "lebih" or class_val == "sedikit lebih":
        return "decrease"
    else:
        return "maintain"

def process_crop_data(crop_name, user_input):
    crop_data = crop_df[crop_df['label'] == crop_name].iloc[0]
    
    if crop_data.empty:
        return f"No data available for crop: {crop_name}"
    
    result = {
        "Crop": crop_name,
    }
    
    factors = ['N', 'P', 'K', 'Temperature', 'Humidity', 'ph']
    for factor in factors:
        user_value = user_input[factor]
        mean_val = crop_data[f'{factor}_mean']
        min_val = crop_data[f'{factor}_min']
        max_val = crop_data[f'{factor}_max']
        
        class_val = determine_class(user_value, min_val, max_val, mean_val)
        color = determine_color(class_val)
        action = determine_action(class_val)
        
        deviation = abs(user_value - mean_val) / mean_val
        
        result[factor] = {
            "class": class_val,
            "colour": color,
            "value": f"{user_value:.2f}",
            "deviation": f"{deviation:.2f}",
            "satuan": "mg" if factor in ['N', 'P', 'K'] else ("%" if factor == "Humidity" else ("pH" if factor == "ph" else "°C")),
            "action": action
        }
    
    return result

def predict_and_recommend(input_data):
    # Prepare input data for prediction
    input_features = np.array([[
        input_data['N'], input_data['P'], input_data['K'],
        input_data['Temperature'], input_data['Humidity'], input_data['ph']
    ]])
    
    # Scale the input data
    scaled_input = scaler.transform(input_features)
    
    # Predict the crop
    predicted_crop_label = stacked_model.predict(scaled_input)
    predicted_crop = label_encoder.inverse_transform(predicted_crop_label)[0]
    
    # Generate recommendations
    recommendations = process_crop_data(predicted_crop, input_data)
    
    return recommendations
# # Save the statistics dictionary
# save_pickle(statistics, 'statistics.pkl')

# Now load the saved statistics to verify
loaded_statistics = load_model('statistics.pkl')
print("Loaded Statistics:", loaded_statistics)

# Example usage
user_input = {
    'N': 90,
    'P': 42,
    'K': 43,
    'Temperature': 20.87,
    'Humidity': 82.00,
    'ph': 6.5
}

# Predict the crop and provide recommendations
recommendations = predict_and_recommend(user_input)

# Print the result
print(json.dumps(recommendations, indent=2))