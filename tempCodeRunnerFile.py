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
from sklearn.exceptions import NotFittedError

# Load the dataset
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/Crop_recommendation.csv'
if not os.path.exists(DATASET_PATH):
    print(f"File not found at: {DATASET_PATH}")
    sys.exit()
crop_data = pd.read_csv(DATASET_PATH)

# Prepare the dataset
y = crop_data['label']
y = y.astype(str)  # Ensure labels are strings
x = crop_data.drop(['label'], axis=1)  # Removed only 'label'

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# --- Added Preprocessing Visualization ---
# 1. Checking for missing data
missing_data = crop_data.isnull().sum()

# 2. Summary statistics
summary_stats = crop_data.describe()

# Visualizations for distribution of numerical features
plt.figure(figsize=(15, 10))

# Creating subplots dynamically based on the number of features
num_columns = len(crop_data.columns) - 1  # Excluding 'label'
num_rows = (num_columns + 2) // 3  # Calculate rows needed for 3 plots per row

plt.figure(figsize=(15, 10))

for i, column in enumerate(crop_data.columns[:-1], 1):  # Exclude the 'label' column
    plt.subplot(num_rows, 3, i)  # Create a grid with enough rows
    sns.histplot(crop_data[column], kde=True)
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()

# Display missing data and summary statistics
print("Missing Data:")
print(missing_data)
print("\nSummary Statistics:")
print(summary_stats)

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    class_labels = label_encoder.classes_
    plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=0)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data):
    plt.figure(figsize=(10, 8))
    
    corr_matrix = data.corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)

    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

plot_correlation_matrix(crop_data.drop('label', axis=1))

# Ensure output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Define parameter grids for each model
param_grids = {
    'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    'DT': {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]},
    'RFC': {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]},
    'GBC': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 5, 7]},
    'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'ANN': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}
}

# Define base models for stacking
base_models = [
    ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
    ('RFC', RandomForestClassifier()),
    ('GBC', GradientBoostingClassifier()),
    ('SVM', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)),
    ('ANN', MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, learning_rate_init=0.001, 
                          early_stopping=True, random_state=42))
]

# Adjust n_splits for StratifiedKFold to prevent the ValueError
class_counts = Counter(y_encoded)
min_class_size = min(class_counts.values())
n_splits = max(5, min_class_size)

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Cross-validation setup
for (name, model), param_grid in zip(base_models, param_grids.values()):
    print(f"\nTuning Hyperparameters for {name} Model...")
    
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    
    y_pred_cv = cross_val_predict(best_model, x_scaled, y_encoded, cv=cv)
    
    accuracy = accuracy_score(y_encoded, y_pred_cv)
    print(f"{name} Model Accuracy: {accuracy:.4f}")

    precision = precision_score(y_encoded, y_pred_cv, average='weighted')
    recall = recall_score(y_encoded, y_pred_cv, average='weighted')
    f1 = f1_score(y_encoded, y_pred_cv, average='weighted')

    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} F1-Score: {f1:.4f}")

    conf_matrix = confusion_matrix(y_encoded, y_pred_cv)
    plot_confusion_matrix(conf_matrix, name)

# Define meta-learner (final estimator)
meta_model = XGBClassifier()

# Define the stacking classifier
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=cv)

# Train and evaluate the stacked model
print("\nTraining Stacked Model with Cross-Validation...")

y_pred_stacked_cv = cross_val_predict(stacked_model, x_scaled, y_encoded, cv=cv)
conf_matrix_stacked = confusion_matrix(y_encoded, y_pred_stacked_cv)

print(f"\nConfusion Matrix for Stacked Model:")
plot_confusion_matrix(conf_matrix_stacked, "Stacked")

# Train the stacked model on the full training set and predict on the test set
stacked_model.fit(x_train, y_train)
y_test_pred = stacked_model.predict(x_test)

accuracy = accuracy_score(y_test, y_test_pred)
print(f"Stacked Model Accuracy: {accuracy:.4f}")

precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Stacked Model Precision: {precision:.4f}")
print(f"Stacked Model Recall: {recall:.4f}")
print(f"Stacked Model F1-Score: {f1:.4f}")

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

save_model(stacked_model, 'stacked')

