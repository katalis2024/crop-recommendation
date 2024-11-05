import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, 
                             f1_score, roc_auc_score)
from sklearn.model_selection import (train_test_split, cross_val_predict, 
                                     StratifiedKFold, GridSearchCV, learning_curve)
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
from sklearn.datasets import make_classification  # Added import for make_classification
from sklearn.metrics import classification_report

# Load the dataset
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/BackupCrop_Recommendation .csv'
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

# Data Visualization
# ... (keep existing visualization code)

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
n_splits = max(2, min(5, min_class_size))  # Ensure at least 2 splits

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Function to evaluate model
def evaluate_model(model, param_grid, x_train, y_train, x_test, y_test, cv):
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred_cv = cross_val_predict(grid_search, x_scaled, y_encoded, cv=cv)
    y_pred_test = best_model.predict(x_test)
    y_pred_proba = best_model.predict_proba(x_test)
    
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    y_pred_proba = best_model.predict_proba(x_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average='weighted')

    conf_matrix = confusion_matrix(y_encoded, y_pred_cv)
    
    return best_model, accuracy, precision, recall, f1, roc_auc, conf_matrix, y_pred_cv

# Evaluate models
model_metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': []}
conf_matrices = []
best_models = {}

for (name, model), param_grid in zip(base_models, param_grids.values()):
    print(f"\nEvaluating {name} Model...")
    best_model, accuracy, precision, recall, f1, roc_auc, conf_matrix, y_pred_cv = evaluate_model(
        model, param_grid, x_train, y_train, x_test, y_test, cv
    )
    
    best_models[name] = best_model
    model_metrics['Model'].append(name)
    model_metrics['Accuracy'].append(accuracy)
    model_metrics['Precision'].append(precision)
    model_metrics['Recall'].append(recall)
    model_metrics['F1 Score'].append(f1)
    model_metrics['ROC AUC'].append(roc_auc)
    conf_matrices.append(conf_matrix)
    
    print(f"{name} Model Results:")
    print(f"Best Parameters: {best_model.get_params()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
   

    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Stacked Model
meta_model = XGBClassifier()
stacked_model = StackingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    final_estimator=meta_model,
    cv=cv
)

print("\nEvaluating Stacked Model...")
stacked_model.fit(x_train, y_train)  # Pastikan model sudah dilatih
y_test_pred = stacked_model.predict(x_test)  # Membuat prediksi pada data uji

# Menghitung metrik evaluasi manual untuk stacked model
accuracy_stacked = accuracy_score(y_test, y_test_pred)
precision_stacked = precision_score(y_test, y_test_pred, average='weighted')
recall_stacked = recall_score(y_test, y_test_pred, average='weighted')
f1_stacked = f1_score(y_test, y_test_pred, average='weighted')
roc_auc_stacked = roc_auc_score(y_test, stacked_model.predict_proba(x_test), multi_class='ovr')

# Menampilkan hasil evaluasi metrik
print("\nStacked Model Evaluation Metrics:")
print(f"Accuracy: {accuracy_stacked:.2f}")
print(f"Precision (weighted): {precision_stacked:.2f}")
print(f"Recall (weighted): {recall_stacked:.2f}")
print(f"F1 Score (weighted): {f1_stacked:.2f}")
print(f"ROC AUC (OVR): {roc_auc_stacked:.2f}")

# Menampilkan classification report
print("\nClassification Report for Stacked Model:")
print(classification_report(y_test, y_test_pred))

# Membuat confusion matrix untuk seluruh kelas crop
conf_matrix_stacked = confusion_matrix(y_test, y_test_pred)
classes = np.unique(y_test)

# Menampilkan confusion matrix secara keseluruhan dalam satu heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for All Crops in Stacked Model')
plt.show()


# Display confusion matrix for stacked model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_stacked, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Stacked Model Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

model_metrics['Model'].append('Stacked Model')
model_metrics['Accuracy'].append(accuracy_stacked)
model_metrics['Precision'].append(precision_stacked)
model_metrics['Recall'].append(recall_stacked)
model_metrics['F1 Score'].append(f1_stacked)
model_metrics['ROC AUC'].append(roc_auc_stacked)
conf_matrices.append(conf_matrix_stacked)

print("Stacked Model Results:")
print(f"Accuracy: {accuracy_stacked:.4f}")
print(f"Precision: {precision_stacked:.4f}, Recall: {recall_stacked:.4f}, F1-Score: {f1_stacked:.4f}")
print(f"ROC AUC: {roc_auc_stacked:.4f}")

# Convert model metrics dictionary to DataFrame for easier plotting
metrics_df = pd.DataFrame(model_metrics)

# Plotting the metrics
metrics_df.set_index('Model').plot(kind='bar', figsize=(14, 8))
plt.title('Evaluation Metrics for Different Models')
plt.ylabel('Scores')
plt.ylim(0, 1)  # Ensure y-axis is between 0 and 1
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(loc='lower right')
plt.show()

# Function to plot confusion matrices for multiple classifiers
def plot_confusion_matrix_comparison(conf_matrices, classifiers, title="Confusion Matrices Comparison"):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))  # Adjust based on the number of classifiers
    axes = axes.flatten()  # Flatten to iterate over axes
    
    for i, (conf_matrix, classifier) in enumerate(zip(conf_matrices, classifiers)):
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
        axes[i].set_title(f'{classifier} Confusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    # Remove extra subplots if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Plot the confusion matrices for comparison
classifiers = [name for name, _ in base_models] + ['Stacked Model']
plot_confusion_matrix_comparison(conf_matrices, model_metrics['Model'])

# Plotting Accuracy Score Comparison for Algorithms
algorithms = [name for name, _ in base_models] + ['Stacked Model']
accuracies = [metrics_df.loc[metrics_df['Model'] == alg, 'Accuracy'].values[0] for alg in algorithms]

plt.figure(figsize=(10, 6))
sns.barplot(x=algorithms, y=accuracies, palette="viridis")
plt.title('Accuracy Score Comparison for Crop Recommendation Algorithms')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)  # Ensure y-axis is between 0 and 1
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Learning Curves
# Generate synthetic data to show learning curves
X_curve, y_curve = make_classification(n_samples=1000, n_features=6, n_informative=4, 
                                       n_redundant=1, n_repeated=0, n_classes=3, 
                                       n_clusters_per_class=2, random_state=42)

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, 
                                                            train_sizes=train_sizes, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean, test_scores_mean, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves for base models and stacked model
# for name, model in best_models.items():
#     plot_learning_curve(model, f"Learning Curve for {name}", X_curve, y_curve, cv=cv)
# plot_learning_curve(best_stacked_model, "Learning Curve for Stacked Model", X_curve, y_curve, cv=cv)

# plt.show()

# Save the stacked model
def save_model(model, model_name):
    with open(os.path.join(output_dir, f'{model_name}_model.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)

# Function to take user input for a new sample and return predictions
def predict_new_sample(x_new, model):
    # Scale the new input sample
    x_new_scaled = scaler.transform([x_new])

    # Get predictions from the Stacked Model
    pred_model = model.predict(x_new_scaled)
    print(f"Predicted Crop: {label_encoder.inverse_transform(pred_model)}")

save_model(stacked_model, 'stacked')

# List input untuk kadar N, P, K, kelembapan, suhu, dan pH
user_inputs = [
    [109.43, 47.23, 19.45, 23.34, 76.23, 6.83],  # Rice
    [18.39, 135.12, 193.45, 23.31, 93.02, 5.83], # Apple
    [41.23, 64.53, 74.99, 20.43, 13.56, 7.37],   # Chickpea
    [100.12, 41.88, 20.21, 22.76, 70.91, 6.71],  # Rice
    [17.97, 140.34, 200.23, 22.13, 89.47, 5.88], # Apple
    [104.52, 45.14, 18.21, 24.89, 73.68, 6.79],  # Rice
    [42.67, 61.53, 73.55, 20.01, 13.75, 7.31],   # Chickpea
    [19.63, 138.88, 201.12, 23.87, 95.13, 5.96], # Apple
    [110.27, 43.12, 18.73, 24.59, 76.13, 6.88],  # Rice
    [38.78, 60.21, 70.34, 19.07, 12.98, 7.27],   # Chickpea
    [20.56, 130.87, 194.55, 22.65, 88.79, 5.81], # Apple
    [105.24, 39.87, 17.32, 22.15, 77.93, 6.85],  # Rice
    [39.66, 60.45, 72.11, 18.72, 14.81, 7.36],   # Chickpea
    [19.80, 134.67, 199.95, 22.59, 92.27, 5.91], # Apple
    [108.44, 40.92, 17.94, 24.12, 74.56, 6.82],  # Rice
    [21.11, 137.64, 205.43, 25.89, 98.23, 6.02], # Apple
    [40.32, 63.11, 75.23, 18.45, 14.23, 7.29],   # Chickpea
    [95.87, 42.67, 19.89, 21.59, 78.99, 6.79],   # Rice
    [19.65, 128.34, 187.55, 20.44, 85.28, 5.90], # Apple
    [37.45, 59.99, 71.12, 17.89, 15.01, 7.20],   # Chickpea
    [98.77, 39.95, 16.98, 22.78, 71.99, 6.66],   # Rice
    [43.21, 63.89, 76.45, 19.33, 16.23, 7.25],   # Chickpea
    [22.37, 132.21, 198.67, 24.67, 90.56, 5.89], # Apple
    [106.98, 44.71, 18.58, 23.94, 72.84, 6.75],  # Rice
    [20.02, 138.75, 202.14, 24.44, 95.78, 5.97], # Apple
    [38.43, 58.21, 69.82, 17.35, 12.56, 7.28],   # Chickpea
    [115.24, 48.56, 20.64, 26.21, 80.12, 6.93],  # Rice
    [18.47, 132.89, 195.32, 23.98, 90.33, 5.85], # Apple
    [41.76, 62.13, 74.23, 20.11, 15.72, 7.32],   # Chickpea
]


# Function to take user input for multiple new samples and return predictions
def predict_new_samples(x_new_list, model):
    predictions = []
    for x_new in x_new_list:
        # Scale the new input sample
        x_new_scaled = scaler.transform([x_new])
        
        # Get predictions from the model
        pred_model = model.predict(x_new_scaled)
        predictions.append(label_encoder.inverse_transform(pred_model))
    
    return predictions

# Get predictions for all inputs
predictions = predict_new_samples(user_inputs, stacked_model)

# Print the predictions
for i, pred in enumerate(predictions):
    print(f"Predicted Crop for Sample {i + 1}: {pred[0]}")  # Assuming single prediction per sample