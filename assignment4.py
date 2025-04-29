import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import recall_score, confusion_matrix, make_scorer, accuracy_score, matthews_corrcoef

# 1. Load the dataset
df = pd.read_csv('/mnt/data/g_data (3).csv', header=None, names=['label','fold','id','sequence'])

# 2. Extract amino acid composition features
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
def aa_composition(seq):
    seq = seq.strip()
    counts = Counter(seq)
    L = len(seq)
    return [counts.get(aa, 0) / L for aa in amino_acids]

X = np.array([aa_composition(s) for s in df['sequence']])
y = df['label'].values

# 3. Define custom specificity scorer (macro-averaged)
def specificity_score(y_true, y_pred):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    FP = cm.sum(axis=0) - np.diag(cm)
    TP = np.diag(cm)
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    spec_per_class = TN / (TN + FP)
    return np.mean(spec_per_class)

# 4. Scorers dictionary
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score, average='macro'),
    'specificity': make_scorer(specificity_score),
    'mcc': make_scorer(matthews_corrcoef)
}

# 5. Cross-validation setup
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# 6. AdaBoost evaluation
ada_results = []
for n in [10, 50, 100, 200, 400]:
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    ada_results.append({
        'n_estimators': n,
        'accuracy': np.mean(scores['test_accuracy']),
        'sensitivity': np.mean(scores['test_sensitivity']),
        'specificity': np.mean(scores['test_specificity']),
        'mcc': np.mean(scores['test_mcc'])
    })
ada_df = pd.DataFrame(ada_results)

# 7. Random Forest evaluation
rf_results = []
for n in [10, 50, 100, 200, 400]:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    rf_results.append({
        'n_estimators': n,
        'accuracy': np.mean(scores['test_accuracy']),
        'sensitivity': np.mean(scores['test_sensitivity']),
        'specificity': np.mean(scores['test_specificity']),
        'mcc': np.mean(scores['test_mcc'])
    })
rf_df = pd.DataFrame(rf_results)

# 8. SVM evaluation
svm_results = []
for kernel in ['linear', 'poly', 'rbf']:
    for C in [0.1, 1, 10, 100]:
        model = SVC(kernel=kernel, C=C)
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        svm_results.append({
            'kernel': kernel,
            'C': C,
            'accuracy': np.mean(scores['test_accuracy']),
            'sensitivity': np.mean(scores['test_sensitivity']),
            'specificity': np.mean(scores['test_specificity']),
            'mcc': np.mean(scores['test_mcc'])
        })
svm_df = pd.DataFrame(svm_results)

# 9. Display results to user
from ace_tools import display_dataframe_to_user
display_dataframe_to_user('AdaBoost Results', ada_df)
display_dataframe_to_user('Random Forest Results', rf_df)
display_dataframe_to_user('SVM Results', svm_df)
