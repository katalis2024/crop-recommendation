import os
import numpy as np
import pandas as pd
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Constants
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/Crop_recommendation.csv'
MODEL1_FILE = 'Stacked_model.pkl'  # File to save model 1
MODEL2_FILE = 'model2_model.pkl'    # File to save model 2

def load_data(dataset_path):
    """Load dataset from a specified path."""
    try:
        return pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit()

def preprocess_data(crop_data):
    """Preprocess the crop data: imputing, encoding, and scaling."""
    imputer = SimpleImputer(strategy='mean')
    x_imputed = imputer.fit_transform(crop_data.drop(['label'], axis=1))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(crop_data['label'])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    return x_scaled, y_encoded, scaler, label_encoder

def tune_hyperparameters(model, X_train, y_train):
    """Tune hyperparameters using RandomizedSearchCV."""
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.3, 0.5, 0.7]
    }
    
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3, n_jobs=-1, 
                                       scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_, random_search.best_params_

def evaluate_model(model, X, y):
    """Evaluate the model performance using regression metrics and visualize results."""
    y_pred = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Visualization
    plt.figure(figsize=(16, 8))

    # Scatter plot of true vs predicted values
    plt.subplot(1, 2, 1)
    plt.scatter(y, y_pred, alpha=0.7, color='b')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.grid(True)

    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y - y_pred
    sns.histplot(residuals, kde=True, color='orange')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Bar plot of evaluation metrics
    metrics = ['MSE', 'RMSE', 'R^2']
    values = [mse, rmse, r2]
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=metrics, y=values, palette='Blues_d')
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def save_model_and_scaler(model, scaler, label_encoder, model_file):
    """Save the trained model, scaler, and label encoder to disk in a single file."""
    with open(model_file, 'wb') as model_file:
        pickle.dump((model, scaler, label_encoder), model_file)

def load_model(model_file):
    """Load model, scaler, and label encoder from a single file."""
    try:
        with open(model_file, "rb") as file:
            model, scaler, label_encoder = pickle.load(file)
        return model, scaler, label_encoder
    except FileNotFoundError:
        print(f"Error: {model_file} not found.")
        sys.exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()

def get_recommendation(N, P, K, Temperature, Humidity, ph, Rainfall, scaler, model1, model2, clusters, mean_values):
    feature_cols = ["N", "P", "K", "Temperature", "Humidity", "ph", "Rainfall"]
    input_data = pd.DataFrame([[N, P, K, Temperature, Humidity, ph, Rainfall]], columns=feature_cols)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data.values)

    # Predict the crop label using the first model (Stacking)
    try:
        final_prediction = model1.predict(input_data_scaled)
        final_prediction_value = final_prediction[0]
    except Exception as e:
        return f"Error during prediction (Model 1): {e}"

    # Convert the numeric prediction to a label
    try:
        predicted_label = label_encoder.inverse_transform([int(final_prediction_value)])[0]
        # Retrieve cluster data for the predicted label to inform model 2
        cluster_data = clusters.loc[predicted_label].values.reshape(1, -1)

        # Predict using Model 2 (XGBRegressor)
        predicted_numeric = model2.predict(cluster_data)
    except Exception as e:
        return f"Error during prediction (Model 2): {e}"

    # Generate the recommendation based on input data
    recommendation = []
    for parameter, value in zip(feature_cols, [N, P, K, Temperature, Humidity, ph, Rainfall]):
        mean_value = mean_values[parameter]
        deviation = value - mean_value
        color = "Green"  # Default color for values within the safe range
        
        # Calculate the percentage of deviation
        if deviation > (mean_value * 0.2):
            color = "Red"
            recommendation.append(f"kadar {parameter} terlalu banyak: {value:.2f} (deviasi: {abs(deviation):.2f} dari {mean_value:.2f}), Warna: {color}")
        elif deviation < -(mean_value * 0.2):
            color = "Red"
            recommendation.append(f"kadar {parameter} terlalu sedikit: {value:.2f} (deviasi: {abs(deviation):.2f} dari {mean_value:.2f}), Warna: {color}")
        elif deviation != 0:
            color = "Orange"
            recommendation.append(f"kadar {parameter} sedikit berbeda: {value:.2f} (deviasi: {abs(deviation):.2f} dari {mean_value:.2f}), Warna: {color}")
        else:
            recommendation.append(f"kadar {parameter} sesuai dengan nilai ideal: {mean_value:.2f}, Warna: hijau")
    
    recommendation.append(f"Tanaman yang disarankan berdasarkan model adalah: {predicted_label} (Output Model 2: {predicted_numeric})")
    
    return "\n".join(recommendation)

if __name__ == "__main__":
    crop_data = load_data(DATASET_PATH)
    X, y, scaler, label_encoder = preprocess_data(crop_data)

    # Assuming clusters and mean_values are defined as follows
    clusters = crop_data.groupby('label').mean()
    mean_values = {col: crop_data[col].mean() for col in crop_data.columns if col != 'label'}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tune hyperparameters for model 1
    model1 = XGBRegressor()
    best_model1, best_params1 = tune_hyperparameters(model1, X_train, y_train)

    # Evaluate the best model
    evaluate_model(best_model1, X_test, y_test)

    # Save the best model, scaler, and label encoder for model 1
    save_model_and_scaler(best_model1, scaler, label_encoder, MODEL1_FILE)

    # Load Model 1
    model1, scaler1, label_encoder1 = load_model(MODEL1_FILE)

    # Initialize and train Model 2 (XGBRegressor)
    model2 = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model2.fit(X_train, y_train)

    # Save Model 2
    with open(MODEL2_FILE, 'wb') as model2_file:
        pickle.dump(model2, model2_file)

    # Example prediction
    result = get_recommendation(120, 60, 80, 30, 50, 6.5, 100, scaler1, model1, model2, clusters, mean_values)
    print(result)
