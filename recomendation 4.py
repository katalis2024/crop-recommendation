import os
import numpy as np
import pandas as pd
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Constants
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/Crop_recommendation.csv'
MODEL1_FILE = 'model1_model.pkl'  # File untuk menyimpan model 1
MODEL2_FILE = 'model2_model.pkl'    # File untuk menyimpan model 2

def load_data(dataset_path):
    """Load dataset from a specified path."""
    try:
        return pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit()

def preprocess_data(crop_data):
    """Preprocess the crop data: imputing, encoding, and scaling."""
    if crop_data.isnull().sum().any():
        imputer = SimpleImputer(strategy='mean')
        x_imputed = imputer.fit_transform(crop_data.drop(['label'], axis=1))
    else:
        x_imputed = crop_data.drop(['label'], axis=1).values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(crop_data['label'])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    return x_scaled, y_encoded, scaler, label_encoder

def create_pipeline():
    """Create a machine learning pipeline with XGBoost."""
    return Pipeline([('regressor', XGBRegressor())])

def tune_hyperparameters(pipeline, X_train, y_train):
    """Tune hyperparameters using RandomizedSearchCV."""
    param_distributions = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 6, 9, 12],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__subsample': [0.5, 0.75, 1.0],
        'regressor__colsample_bytree': [0.3, 0.5, 0.7]
    }
    
    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=20, cv=3, n_jobs=-1, 
                                       scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_, random_search.best_params_

def evaluate_model(model, X, y):
    """Evaluate the model performance using regression metrics."""
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.grid()
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

    # Predict the crop label using the first model (XGBoost)
    try:
        final_prediction = model1.predict(input_data_scaled)
        final_prediction_value = final_prediction[0]
    except Exception as e:
        return f"Error during prediction (Model 1): {e}"

    # Convert the numeric prediction to a label
    try:
        predicted_label = label_encoder.inverse_transform([int(final_prediction_value)])[0]  # Convert to int if necessary
        # Retrieve cluster data for the predicted label to inform model 2
        cluster_data = clusters.loc[predicted_label].values.reshape(1, -1)

        # Ensure that the cluster_data is a valid input for model2
        if model2 is not None:
            predicted_numeric = model2.predict(cluster_data)
        else:
            predicted_numeric = "Model 2 tidak tersedia"
    except Exception as e:
        return f"Error during prediction (Model 2): {e}"

    # Generate the recommendation based on input data
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
    
    recommendation.append(f"Tanaman yang disarankan berdasarkan model adalah: {predicted_label} (Output Model 2: {predicted_numeric})")
    
    return "\n".join(recommendation)

if __name__ == "__main__":
    crop_data = load_data(DATASET_PATH)
    X, y, scaler, label_encoder = preprocess_data(crop_data)

    # Assuming clusters and mean_values are defined as follows
    clusters = crop_data.groupby('label').mean()
    mean_values = {col: crop_data[col].mean() for col in crop_data.columns if col != 'label'}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1
    pipeline = create_pipeline()
    best_model1, best_params1 = tune_hyperparameters(pipeline, X_train, y_train)

    print("Best parameters for Model 1:", best_params1)
    evaluate_model(best_model1, X_test, y_test)

    save_model_and_scaler(best_model1, scaler, label_encoder, MODEL1_FILE)

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


    # Load Model 1
    model1, scaler, label_encoder = load_model(MODEL1_FILE)

    # Model 2 (Jika ada, proses serupa dilakukan)
    model2 = None  # Ganti ini dengan pemuatan model2 jika ada

    # Contoh menggunakan fungsi get_recommendation
    recommendation_output = get_recommendation(
        N=60, 
        P=30, 
        K=50, 
        Temperature=25, 
        Humidity=70, 
        ph=6.5, 
        Rainfall=100, 
        scaler=scaler, 
        model1=model1, 
        model2=model2, 
        clusters=clusters, 
        mean_values=mean_values
    )
    
    print(recommendation_output)
