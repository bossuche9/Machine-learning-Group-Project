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

INPUT_CSV = "g_data.csv"     # input file
SEQ_COL   = "sequence"       # protein sequence
LABEL_COL = "class"          # labels ('Fold1','Fold2'...)
TEST_SIZE = 0.20
RANDOM_SEED = 42

# ———  LOAD DATA ———
print("Loading data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path1 = os.path.join(script_dir, INPUT_CSV)
df = pd.read_csv(csv_path1, header=None, names=['id', 'class', 'protein_id', 'sequence'])
sequences = df[SEQ_COL].astype(str)
labels    = df[LABEL_COL].astype(str)

# ——— FEATURE EXTRACTION ———
print("Extracting features...")
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def extract_features(seq: str):
    seq = seq.upper()
    L = len(seq)
    counts = [seq.count(n) for n in amino_acids]
    comps  = [cnt / L for cnt in counts]
    return counts + comps + [L]

X = np.array([extract_features(seq) for seq in sequences])
y = labels.values


# ———  TRAIN/TEST SPLIT & SCALE ———
print("Splitting and scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    stratify=y, random_state=RANDOM_SEED
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ———  DEFINE CLASSIFIERS ———
classifiers = {
    "KNN"    : KNeighborsClassifier(n_neighbors=5),
    "SVM"    : SVC(kernel='rbf', C=1.0, probability=False),
    "NB"     : GaussianNB(),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "MLP"    : MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=RANDOM_SEED)
}

# Add AdaBoost and Random forest with multiple learners
for n in [10, 50, 100, 200, 400]:
    classifiers[f"AdaBoost_{n}"] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=RANDOM_SEED
    )
    classifiers[f"RF_{n}"] = RandomForestClassifier(
        n_estimators=n,
        random_state=RANDOM_SEED
    )

# ——— ) TRAIN, EVALUATE, CROSS-VAL ———
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

# print summary
print("\n-=========- SUMMARY -=========-")
for name, res in results.items():
    print(f"{name}: test={res['test_acc']:.4f}, CV={res['cv_mean']:.4f} ± {res['cv_std']:.4f}")