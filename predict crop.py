import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, 
                             f1_score, roc_auc_score)
from sklearn.model_selection import (train_test_split, cross_val_predict, 
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              StackingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
DATASET_PATH = 'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app\data\BackupCrop_Recommendation .csv'
if not os.path.exists(DATASET_PATH):
    print(f"File not found at: {DATASET_PATH}")
    sys.exit()
crop_data = pd.read_csv(DATASET_PATH)

# Prepare the dataset by removing 'rainfall'
crop_data = crop_data.drop(['Rainfall'], axis=1)

y = crop_data['label'].astype(str)
x = crop_data.drop(['label'], axis=1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# --- Data Visualization --- 
# 1. Checking for missing data
missing_data = crop_data.isnull().sum()
print("Missing Data:\n", missing_data)

# 2. Summary statistics
summary_stats = crop_data.describe()
print("\nSummary Statistics:\n", summary_stats)

# 3. Distribution of numerical features (excluding 'rainfall')
plt.figure(figsize=(15, 10))
num_columns = len(crop_data.columns) - 1
num_rows = (num_columns + 2) // 3
for i, column in enumerate(crop_data.columns[:-1], 1):  # Exclude 'label'
    plt.subplot(num_rows, 3, i)
    sns.histplot(crop_data[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Correlation Matrix
def plot_correlation_matrix(data):
    plt.figure(figsize=(10, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

plot_correlation_matrix(crop_data.drop('label', axis=1))

# Output directory for saving models
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Parameter grids for each model
param_grids = {
    'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    'DT': {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]},
    'RFC': {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]},
    'GBC': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 5, 7]},
    'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'ANN': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}
}

# Base models for stacking
base_models = [
    ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
    ('RFC', RandomForestClassifier()),
    ('GBC', GradientBoostingClassifier()),
    ('SVM', SVC(kernel='rbf', probability=True, random_state=42)),
    ('ANN', MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, learning_rate_init=0.001, 
                          early_stopping=True, random_state=42))
]

# Adjust StratifiedKFold based on the smallest class size
class_counts = Counter(y_encoded)
min_class_size = min(class_counts.values())
n_splits = min(5, min_class_size)

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Cross-validation and hyperparameter tuning
for (name, model), param_grid in zip(base_models, param_grids.values()):
    print(f"\nTuning Hyperparameters for {name} Model...")
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    
    y_pred_cv = cross_val_predict(best_model, x_scaled, y_encoded, cv=cv)
    
    accuracy = accuracy_score(y_encoded, y_pred_cv)
    precision = precision_score(y_encoded, y_pred_cv, average='weighted')
    recall = recall_score(y_encoded, y_pred_cv, average='weighted')
    f1 = f1_score(y_encoded, y_pred_cv, average='weighted')
    
    print(f"{name} Model Accuracy: {accuracy:.4f}")
    print(f"{name} Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Confusion matrix plot
    conf_matrix = confusion_matrix(y_encoded, y_pred_cv)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Meta-learner and Stacking model
meta_model = XGBClassifier()
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=cv)

# Cross-validation on stacked model
y_pred_stacked_cv = cross_val_predict(stacked_model, x_scaled, y_encoded, cv=cv)
conf_matrix_stacked = confusion_matrix(y_encoded, y_pred_stacked_cv)
print(f"\nConfusion Matrix for Stacked Model:")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_stacked, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Stacked Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Final training on the stacked model
stacked_model.fit(x_train, y_train)
y_test_pred = stacked_model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))

# Calculate accuracy per crop class
unique_classes = np.unique(y_test)
class_accuracies = {}
for cls in unique_classes:
    # Mendapatkan indeks untuk setiap kelas (benar dan prediksi)
    idx = np.where(y_test == cls)
    class_accuracies[cls] = accuracy_score(y_test[idx], y_test_pred[idx])

print("\nAccuracy per Crop Class:")
for crop_class, accuracy in class_accuracies.items():
    print(f"{crop_class}: {accuracy:.2f}")

# Stacked model evaluation on test set
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Stacked Model Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# ROC-AUC for the stacked model
if hasattr(stacked_model, "predict_proba"):
    y_proba = stacked_model.predict_proba(x_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average='weighted')
    print(f"Stacked Model ROC-AUC: {roc_auc:.4f}")
else:
    print("Stacked Model does not support probability estimation for ROC-AUC.")

# Save the stacked model
def save_model(model, model_name):
    with open(os.path.join(output_dir, f'{model_name}_model.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)

# Save the scaler and label encoder
def save_pickle_object(obj, obj_name):
    with open(os.path.join(output_dir, f'{obj_name}_fixmodel.pkl'), 'wb') as obj_file:
        pickle.dump(obj, obj_file)


# Function to take user input for multiple new samples and return predictions
def predict_new_samples(x_new_list, model):
    # Iterate over each new input sample
    for i, x_new in enumerate(x_new_list):
        # Scale the new input sample
        x_new_scaled = scaler.transform([x_new])

        # Get predictions from the Stacked Model
        pred_model = model.predict(x_new_scaled)
        predicted_crop = label_encoder.inverse_transform(pred_model)
        
        # Output the result for each sample
        print(f"Sample {i+1}: Input = {x_new}")
        print(f"Predicted Crop: {predicted_crop[0]}")
        print("-" * 30)

# List input untuk kadar N, P, K, kelembapan, suhu, dan pH
user_inputs = [
    [117.77, 46.24, 19.56, 23.99, 79.84, 6.91],  # Cotton
    ]

# Make predictions for all user inputs
predict_new_samples(user_inputs, stacked_model)
