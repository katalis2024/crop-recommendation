import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, 
                             f1_score, roc_auc_score)
from sklearn.model_selection import (train_test_split, cross_val_predict, 
                                       StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              StackingClassifier)
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
y = y.astype(str)  # Ensure labels are strings
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

# Train and evaluate the stacked model
print("\nTraining Stacked Model...")

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store cross-validation results
confusion_matrices = {}
model_predictions = {}
model_accuracies = {}

# Create output directory
output_dir = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/confusion_matrices/'
os.makedirs(output_dir, exist_ok=True)

# Train and evaluate the stacked model
print(f"\nTraining Stacked Model...")
y_pred = cross_val_predict(stacked_model, x_scaled, y_encoded, cv=cv)
conf_matrix = confusion_matrix(y_encoded, y_pred)
confusion_matrices["Stacked"] = conf_matrix

print(f"\nConfusion Matrix for Stacked Model:")
print(conf_matrix)

# Plot confusion matrix
plot_confusion_matrix(conf_matrix, "Stacked", output_dir)

# Train the stacked model on the full training set and predict on the test set
stacked_model.fit(x_train, y_train)
y_test_pred = stacked_model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)  # Ensure y_test is y_encoded
print(f"Stacked Model Accuracy on Test Set: {accuracy:.4f}")

# Store predictions and accuracies
model_predictions["Stacked"] = y_test_pred
model_accuracies["Stacked"] = accuracy

# Precision, Recall, F1-Score
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Stacked Model Precision: {precision:.4f}")
print(f"Stacked Model Recall: {recall:.4f}")
print(f"Stacked Model F1-Score: {f1:.4f}")

# ROC-AUC (if model supports probability estimation)
if hasattr(stacked_model, "predict_proba"):
    y_proba = stacked_model.predict_proba(x_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average='weighted')
    print(f"Stacked Model ROC-AUC: {roc_auc:.4f}")
else:
    print("Stacked Model does not support probability estimation for ROC-AUC.")

# Save the stacked model
save_model(stacked_model, "Stacked")

# Function to take user input for a new sample and return predictions
def predict_new_sample(x_new, model):
    # Scale the new input sample
    x_new_scaled = scaler.transform([x_new])

    # Get predictions from the Stacked Model
    pred_model = model.predict(x_new_scaled)
    print(f"Model Prediction: {label_encoder.inverse_transform(pred_model)}")

# Example usage for prediction (you can replace this with actual user input)
new_sample = [60, 20, 30, 75, 5.5, 90, 20]  # Example new input sample
predict_new_sample(new_sample, stacked_model)