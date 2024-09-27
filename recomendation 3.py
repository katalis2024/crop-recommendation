import os
import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load the dataset
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/Crop_recommendation.csv'

if os.path.exists(DATASET_PATH):
    crop_data = pd.read_csv(DATASET_PATH)
else:
    print(f"File not found at: {DATASET_PATH}")
    sys.exit()

# Inspect the dataset
print(crop_data.head())
print(crop_data.dtypes)

# Prepare the dataset
y = crop_data['label']
x = crop_data.drop(['label'], axis=1)

# Convert all columns to numeric, coerce errors to NaN, and then drop those rows
x = x.apply(pd.to_numeric, errors='coerce')
x = x.dropna()  # Drop rows with NaN values
y = y.loc[x.index]  # Align y with x after dropping NaN rows

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the data using StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Initialize the XGBClassifier
xgb_model = XGBClassifier()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Evaluate the model with cross-validation
scores = cross_val_score(best_model, x_scaled, y_encoded, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Train the model with the best parameters
best_model.fit(X_train, y_train)

# Save the model and scaler
with open('xgb_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Define the clusters
clusters = crop_data.groupby("label").mean()
clusters = clusters.apply(pd.to_numeric, errors='coerce')

# Define the recommendation function
def get_recommendation(N, P, K, Temperature, Humidity, ph, Rainfall):
    # Load the scaler and model
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    with open('xgb_model.pkl', 'rb') as model_file:
        xgb_model = pickle.load(model_file)
    
    # Create a DataFrame for the input with feature names
    feature_cols = ["N", "P", "K", "Temperature", "Humidity", "ph", "Rainfall"]
    input_data = pd.DataFrame([[N, P, K, Temperature, Humidity, ph, Rainfall]], columns=feature_cols)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the crop label using the XGBoost model
    try:
        predicted_label_encoded = xgb_model.predict(input_data_scaled)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
    except Exception as e:
        return f"Error during prediction: {e}"
    
    # Get the ideal ranges for the predicted crop
    if predicted_label in clusters.index:
        ideal_ranges = clusters.loc[predicted_label]
    else:
        return f"Error: The predicted crop '{predicted_label}' is not available in the dataset."
    
    # Define the ideal range for each parameter
    range_params = {
        "N": (20, 60),
        "P": (10, 50),
        "K": (30, 80),
        "Temperature": (15, 35),
        "Humidity": (40, 80),
        "ph": (5.5, 7.0),
        "Rainfall": (100, 300)
    }
    
    # Create a recommendation string based on the comparison between input values and ideal ranges
    recommendation = []
    for parameter, value in zip(feature_cols, [N, P, K, Temperature, Humidity, ph, Rainfall]):
        min_range, max_range = range_params[parameter]
        if value < min_range:
            recommendation.append(f"{parameter} kekurangan, silakan tambahkan.")
        elif value > max_range:
            recommendation.append(f"{parameter} kelebihan, silakan kurangi.")
        else:
            recommendation.append(f"{parameter} sesuai dengan standar nasional.")
    
    # Add information about the suggested crop from the model
    recommendation.append(f"Tanaman yang disarankan berdasarkan model adalah: {predicted_label}")
    
    return "\n".join(recommendation)

# Test the recommendation function
print(get_recommendation(90, 40, 40, 20, 80, 7, 200))

# Evaluate the model performance
y_pred = best_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate ROC curve and AUC
y_prob = best_model.predict_proba(X_test)  # Probability estimates for the test set
n_classes = len(label_encoder.classes_)

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
