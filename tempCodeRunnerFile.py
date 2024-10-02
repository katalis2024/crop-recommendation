import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, 
                             f1_score, roc_auc_score)
from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict, 
                                       StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Load the dataset
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/Crop_recommendation.csv'
if not os.path.exists(DATASET_PATH):
    print(f"File not found at: {DATASET_PATH}")
    sys.exit()
crop_data = pd.read_csv(DATASET_PATH)

# Prepare the dataset
y = crop_data['label']

# Check unique labels and their types
unique_labels = y.unique()
print("Unique labels and their types:")
for label in unique_labels:
    print(f"Label: {label}, Type: {type(label)}")

# Convert labels to strings (or integers if appropriate)
y = y.astype(str)  # Convert to string type

# Proceed with dropping labels and encoding
x = crop_data.drop(['label'], axis=1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
clusters = hc.fit_predict(x_scaled)

# Modular function to save model
def save_model(model, model_name):
    model_filename = f'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/{model_name}_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"{model_name} has been saved to {model_filename}")

# Modular function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Confusion matrix for {model_name} saved as {plot_filename}")

# Modular function to visualize clustering results
def plot_dendrogram(x_scaled):
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(x_scaled, method='ward'))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

# Plot the dendrogram
plot_dendrogram(x_scaled)

# Analyze clustering results
cluster_df = pd.DataFrame({'Cluster': clusters, 'Label': y})
for i in range(4):
    print(f"Crops in Cluster {i}: {cluster_df[cluster_df['Cluster'] == i]['Label'].unique()}")
    print("---------------------------------------------------------------")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

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

# Define meta-learner (final estimator)
meta_model = XGBClassifier()

# Define the stacking classifier
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=StratifiedKFold(n_splits=5))

# Train and evaluate models
models = {
    "Stacked": stacked_model,  # Stacked classifier
    "XGB": XGBClassifier()     # XGBoost as standalone for comparison
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store cross-validation results
cv_results = {}
confusion_matrices = {}
model_predictions = {}
model_accuracies = {}

# Create output directory
output_dir = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/confusion_matrices/'
os.makedirs(output_dir, exist_ok=True)

# Train, evaluate, and pickle each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Cross-validation scores
    cv_scores = cross_val_score(model, x_scaled, y_encoded, cv=cv, scoring='accuracy')
    mean_accuracy = np.mean(cv_scores)
    std_dev = np.std(cv_scores)

    print(f"{model_name} Mean Accuracy: {mean_accuracy:.4f}")
    print(f"{model_name} Standard Deviation: {std_dev:.4f}")

    # Cross-validation predictions
    y_pred = cross_val_predict(model, x_scaled, y_encoded, cv=cv)
    conf_matrix = confusion_matrix(y_encoded, y_pred)
    confusion_matrices[model_name] = conf_matrix

    print(f"\nConfusion Matrix for {model_name}:")
    print(conf_matrix)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, model_name, output_dir)

    # Train the model on the full training set and predict on the test set
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)

    # Decode predictions
    y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test_decoded, y_test_pred_decoded)
    print(f"{model_name} Accuracy on Test Set: {accuracy:.4f}")

    # Store predictions and accuracies
    model_predictions[model_name] = y_test_pred
    model_accuracies[model_name] = accuracy

    # Precision, Recall, F1-Score
    precision = precision_score(y_test_decoded, y_test_pred_decoded, average='weighted')
    recall = recall_score(y_test_decoded, y_test_pred_decoded, average='weighted')
    f1 = f1_score(y_test_decoded, y_test_pred_decoded, average='weighted')

    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1-Score: {f1:.4f}")

    # ROC-AUC (if model supports probability estimation)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average='weighted')
        print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    else:
        print(f"{model_name} does not support probability estimation for ROC-AUC.")

    # Save the model
    save_model(model, model_name)   

# Store the evaluation metrics for each model
model_metrics = {
    "Model": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "Accuracy": [],
}

# Collect all model metrics
for model_name in models:
    precision = precision_score(y_test_decoded, model_predictions[model_name], average='weighted')
    recall = recall_score(y_test_decoded, model_predictions[model_name], average='weighted')
    f1 = f1_score(y_test_decoded, model_predictions[model_name], average='weighted')
    accuracy = model_accuracies[model_name]
    
    model_metrics['Model'].append(model_name)
    model_metrics['Precision'].append(precision)
    model_metrics['Recall'].append(recall)
    model_metrics['F1-Score'].append(f1)
    model_metrics['Accuracy'].append(accuracy)

# Add stacking metrics
stacking_precision = precision_score(y_test_decoded, model_predictions['Stacked'], average='weighted')
stacking_recall = recall_score(y_test_decoded, model_predictions['Stacked'], average='weighted')
stacking_f1 = f1_score(y_test_decoded, model_predictions['Stacked'], average='weighted')
stacking_accuracy = model_accuracies['Stacked']

model_metrics['Model'].append('Stacked')
model_metrics['Precision'].append(stacking_precision)
model_metrics['Recall'].append(stacking_recall)
model_metrics['F1-Score'].append(stacking_f1)
model_metrics['Accuracy'].append(stacking_accuracy)

# Convert metrics to DataFrame for visualization
metrics_df = pd.DataFrame(model_metrics)
print("\nModel Metrics:")
print(metrics_df)

# Save the metrics DataFrame to CSV
metrics_df.to_csv('C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/model_metrics.csv', index=False)
print("Model metrics saved to CSV.")

