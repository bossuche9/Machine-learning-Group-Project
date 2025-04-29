# -*- coding: utf-8 -*-
"""Machine Learning Project Script

This script performs protein subcellular localization prediction using
various classification models on pre-computed features.

Steps:
1. Load data from 'comp_occur.csv'.
2. Split data into training and testing sets.
3. Define classification models (KNN, SVC, RandomForest).
4. Create pipelines including StandardScaler for each model.
5. Evaluate models using:
    a) 5-fold cross-validation (with scaling within folds).
    b) Performance metrics on the independent test set.
"""

# %% Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# %% Configuration
DATA_FILE = "comp_occur.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 0
CV_FOLDS = 5

# %% Load Data
print("Loading data...")
dataset = pd.read_csv(DATA_FILE)
X = dataset.iloc[:, 1:].values  # Features
y_str = dataset.iloc[:, 0].values   # Labels (protein location) - Original string labels
print(f"Dataset shape: {dataset.shape}")
print(f"Features shape: {X.shape}")
print(f"Original Labels shape: {y_str.shape}")
print(f"Unique original labels: {np.unique(y_str)}")

# Encode string labels to numerical labels
le = LabelEncoder()
y = le.fit_transform(y_str) # Encode labels
print(f"Encoded Labels shape: {y.shape}")
print(f"Unique encoded labels: {np.unique(y)}")
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# %% Split Data
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}\n")

# %% Define Models and Pipelines
print("Defining models and pipelines...")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=18, metric='minkowski', p=2.5)
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', knn)
])

# Support Vector Classifier
svc = SVC(kernel='rbf', random_state=RANDOM_STATE, decision_function_shape='ovo', probability=True) # probability=True for potential future use
pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', svc)
])

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=10000, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1) # Use all available CPU cores
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', rf)
])

models = {
    "KNN": pipe_knn,
    "SVC": pipe_svc,
    "RandomForest": pipe_rf
}

# %% Train and Evaluate Models
print("Training and evaluating models...")

for name, model_pipeline in models.items():
    print(f"--- Evaluating {name} ---")

    # --- Cross-Validation ---
    print("Running 5-fold cross-validation...")
    # Perform cross-validation on the *training* data with the pipeline
    # This ensures scaling is done correctly within each fold
    cv_scores = cross_val_score(estimator=model_pipeline,
                                X=X_train,  # Use training data for CV
                                y=y_train,
                                cv=CV_FOLDS,
                                scoring='accuracy', # Can specify other metrics
                                n_jobs=-1)

    print(f"CV Accuracy ({CV_FOLDS}-fold): {cv_scores.mean()*100:.2f} % (+/- {cv_scores.std()*100:.2f} %)")

    # --- Test Set Evaluation ---
    print("Evaluating on the independent test set...")
    # Fit the pipeline on the full training data
    model_pipeline.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model_pipeline.predict(X_test)

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Test Set Accuracy: {test_accuracy*100:.2f} %")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print("-------------------------\n")

print("Script finished.")
