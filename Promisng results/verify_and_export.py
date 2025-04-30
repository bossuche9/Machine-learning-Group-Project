#!/usr/bin/env python3
"""
verify_and_export.py

1. Load your g_data.csv (assumes a column 'sequence' and a column 'class' with your labels).
2. Extract amino-acid composition features.
3. Export features.csv (for Weka) and features.arff.
4. Train/test split, standardize.
5. Train KNN, SVM, NB, RF, Bagging, MLP.
6. Print Accuracy, Confusion Matrix, Classification Report.
7. Compute Sensitivity, Specificity, MCC per class.
8. 5-fold cross-validation accuracy.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    multilabel_confusion_matrix, matthews_corrcoef
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree     import DecisionTreeClassifier


# ——— CONFIG 

INPUT_CSV = "g_data.csv"
SEQ_COL   = "sequence"
LABEL_COL = "class"
TEST_SIZE = 0.20
RANDOM_SEED = 42

# ——— 1) LOAD DATA ———
print("Loading data…")
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path1 = os.path.join(script_dir, INPUT_CSV)
df = pd.read_csv(csv_path1, header=None, names=['id', 'class', 'protein_id', 'sequence'])
sequences = df[SEQ_COL].astype(str)
labels    = df[LABEL_COL].astype(str)

# ——— 2) FEATURE EXTRACTION ———
print("Extracting features…")
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def extract_features(seq: str):
    seq = seq.upper()
    L = len(seq)
    counts = [seq.count(aa) for aa in amino_acids]
    comps  = [cnt / L for cnt in counts]
    return counts + comps + [L]

X = np.array([extract_features(seq) for seq in sequences])
y = labels.values

# ——— 3) EXPORT FOR WEKA ———
print("Exporting features.csv and features.arff for Weka…")
# 3a) CSV
col_names = [f"count_{aa}" for aa in amino_acids] + \
            [f"comp_{aa}"  for aa in amino_acids] + \
            ["seq_length", LABEL_COL]
out_df = pd.DataFrame(np.column_stack((X, y)), columns=col_names)
out_df.to_csv("features.csv", index=False)

# 3b) ARFF
with open("features.arff", "w") as f:
    f.write("@RELATION gram_positive\n\n")
    for aa in amino_acids:
        f.write(f"@ATTRIBUTE count_{aa} NUMERIC\n")
    for aa in amino_acids:
        f.write(f"@ATTRIBUTE comp_{aa}  NUMERIC\n")
    f.write("@ATTRIBUTE seq_length NUMERIC\n")
    classes = sorted(df[LABEL_COL].unique())
    cls_list = ",".join(classes)
    f.write(f"@ATTRIBUTE {LABEL_COL} {{{cls_list}}}\n\n")
    f.write("@DATA\n")
    for row in out_df.itertuples(index=False):
        f.write(",".join(map(str, row)) + "\n")

# ——— 4) TRAIN/TEST SPLIT & SCALE ———
print("Splitting and scaling…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    stratify=y, random_state=RANDOM_SEED
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ——— 5) DEFINE CLASSIFIERS ———
classifiers = {
    "KNN"    : KNeighborsClassifier(n_neighbors=5),
    "SVM"    : SVC(kernel='rbf', C=1.0, probability=False),
    "NB"     : GaussianNB(),
    "RF"     : RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=RANDOM_SEED),
    "MLP"    : MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=RANDOM_SEED)
}

# ——— 6–8) TRAIN, EVALUATE, CROSS-VAL ———
results = {}
for name, clf in classifiers.items():
    print(f"\n=== {name} ===")
    # train
    clf.fit(X_train_s, y_train)
    # predict
    y_pred = clf.predict(X_test_s)
    # accuracy & confusion
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (test): {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # per-class Sensitivity, Specificity, MCC
    print("Per-class Sensitivity, Specificity, MCC:")
    mcm = multilabel_confusion_matrix(y_test, y_pred, labels=clf.classes_)
    for cls, cm in zip(clf.classes_, mcm):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc  = matthews_corrcoef(
            (y_test == cls).astype(int),
            (y_pred == cls).astype(int)
        ) if ((tp+fn)>0 and (tp+fp)>0 and (tn+fn)>0 and (tn+fp)>0) else 0.0
        print(f"  {cls}:  Sens={sens:.3f}, Spec={spec:.3f}, MCC={mcc:.3f}")

    # 5-fold CV on training set
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=kf, scoring='accuracy')
    print(f"CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results[name] = {
        "test_acc": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std" : cv_scores.std()
    }

# ——— SUMMARY ———
print("\n=== SUMMARY ===")
for name, res in results.items():
    print(f"{name}: test={res['test_acc']:.4f}, CV={res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
