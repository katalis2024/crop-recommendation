import os
import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


# Constants
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/Crop_recommendation.csv'
MODEL_FILE = 'xgb_model.pkl'
SCALER_FILE = 'scaler.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'

def load_data(dataset_path):
    """Load dataset from a specified path."""
    try:
        return pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit()

def preprocess_data(crop_data):
    """Preprocess the crop data: imputing, encoding, and scaling."""
    # Check for missing values
    if crop_data.isnull().sum().any():
        imputer = SimpleImputer(strategy='mean')
        x_imputed = imputer.fit_transform(crop_data.drop(['label'], axis=1))
    else:
        x_imputed = crop_data.drop(['label'], axis=1).values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(crop_data['label'])

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    return x_scaled, y_encoded, scaler, label_encoder

def create_pipeline():
    """Create a machine learning pipeline with XGBoost."""
    return Pipeline([('classifier', XGBClassifier())])

def tune_hyperparameters(pipeline, X_train, y_train):
    """Tune hyperparameters using RandomizedSearchCV."""
    param_distributions = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 6, 9, 12],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.5, 0.75, 1.0],
        'classifier__colsample_bytree': [0.3, 0.5, 0.7]
    }
    
    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=20, cv=3, n_jobs=-1, scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_, random_search.best_params_

def evaluate_model(model, X, y, label_encoder):
    """Evaluate the model performance using various metrics."""
    y_pred = model.predict(X)
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred))

    # Calculate ROC curve and AUC for multi-class classification (macro-average)
    y_prob = model.predict_proba(X)
    y_bin = label_binarize(y, classes=np.arange(len(label_encoder.classes_)))
    roc_auc = roc_auc_score(y_bin, y_prob, average='macro')
    print(f"Macro-Average AUC: {roc_auc:.2f}")

    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def save_model_and_scaler(model, scaler, label_encoder):
    """Save the trained model, scaler, and label encoder to disk."""
    with open(MODEL_FILE, 'wb') as model_file:
        pickle.dump(model, model_file)

    with open(SCALER_FILE, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
        
    with open(LABEL_ENCODER_FILE, 'wb') as le_file:
        pickle.dump(label_encoder, le_file)

def get_recommendation(N, P, K, Temperature, Humidity, ph, Rainfall, scaler, model, label_encoder, clusters, mean_values):
    """Get crop recommendations based on input parameters."""
    feature_cols = ["N", "P", "K", "Temperature", "Humidity", "ph", "Rainfall"]
    input_data = pd.DataFrame([[N, P, K, Temperature, Humidity, ph, Rainfall]], columns=feature_cols)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the crop label using the XGBoost model
    try:
        predicted_label_encoded = model.predict(input_data_scaled)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
    except Exception as e:
        return f"Error during prediction: {e}"
    
    # Get the ideal ranges for the predicted crop
    if predicted_label in clusters.index:
        ideal_ranges = clusters.loc[predicted_label]
    else:
        return f"Error: The predicted crop '{predicted_label}' is not available in the dataset."
    
    # Create a recommendation string based on the comparison between input values and ideal ranges
    recommendation = []
    for parameter, value in zip(feature_cols, [N, P, K, Temperature, Humidity, ph, Rainfall]):
        mean_value = mean_values[parameter]
        deviation = value - mean_value
        color = "Green"  # Default color for values within the safe range
        
        if deviation > (mean_value * 0.2):
            color = "Red"
            recommendation.append(f"kadar {parameter} terlalu banyak {abs(deviation):.2f} poin dari yang seharusnya {mean_value:.2f}, Warna: {color}")
        elif deviation < -(mean_value * 0.2):
            color = "Red"
            recommendation.append(f"kadar {parameter} terlalu sedikit {abs(deviation):.2f} poin dari yang seharusnya {mean_value:.2f}, Warna: {color}")
        elif deviation != 0:
            color = "Orange"
            recommendation.append(f"kadar {parameter} sedikit berbeda {abs(deviation):.2f} poin dari yang seharusnya {mean_value:.2f}, Warna: {color}")
        else:
            recommendation.append(f"kadar {parameter} sesuai dengan nilai ideal {mean_value:.2f}, Warna: hijau")
    
    # Add information about the suggested crop from the model
    recommendation.append(f"Tanaman yang disarankan berdasarkan model adalah: {predicted_label}")
    
    return "\n".join(recommendation)

# Main execution flow
crop_data = load_data(DATASET_PATH)
x_scaled, y_encoded, scaler, label_encoder = preprocess_data(crop_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Create and tune the model pipeline
pipeline = create_pipeline()
best_model, best_params = tune_hyperparameters(pipeline, X_train, y_train)
print("Best parameters:", best_params)

# Evaluate the model
evaluate_model(best_model, X_test, y_test, label_encoder)

# Save the model, scaler, and label encoder
save_model_and_scaler(best_model, scaler, label_encoder)

# Define clusters and mean values for recommendations
clusters = crop_data.groupby("label").mean()
clusters = clusters.apply(pd.to_numeric, errors='coerce')
mean_values = clusters.mean().to_dict()

# Test the recommendation function
print(get_recommendation(90, 40, 40, 20, 80, 7, 200, scaler, best_model, label_encoder, clusters, mean_values))
