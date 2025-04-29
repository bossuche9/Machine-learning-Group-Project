import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load g_data.csv (Gram-positive proteins)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "g_data.csv")
g_data = pd.read_csv(csv_path)

# Step 2: Extract sequences and labels
# Sequence is in column index 3, label is 'Fold1'
sequences = g_data.iloc[:, 3]
labels = g_data['Fold1']

# Step 3: Feature Extraction (Occurrence + Composition)
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def extract_features(seq):
    seq = str(seq).upper()
    length = len(seq)
    features = [seq.count(aa) for aa in amino_acids]  # Occurrence
    features += [seq.count(aa) / length for aa in amino_acids]  # Composition (normalized)
    features.append(length)  # Add sequence length as feature
    return features

X = np.array([extract_features(seq) for seq in sequences])
y = labels.values

# Step 4: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Define classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
    "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# Step 7: Train and Evaluate
results = {}
for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (Test Set): {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-Validation (5-fold)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Save results
    results[name] = {
        "test_accuracy": acc,
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std()
    }

# Step 8: Summary of Results
print("\n--- Summary ---")
for name, res in results.items():
    print(f"{name}: Test Acc = {res['test_accuracy']:.4f}, CV Acc = {res['cv_mean_accuracy']:.4f} (+/- {res['cv_std_accuracy']:.4f})")
