"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset
and saves:
  - Trained models (pickle files) in model/
  - A fitted StandardScaler (model/scaler.pkl)
  - test_data.csv (held-out test split, used by the Streamlit app)
  - metrics_summary.csv (comparison table used in README.md)

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source : UCI ML Repository / built into scikit-learn (sklearn.datasets.load_breast_cancer)
Task   : Binary classification (malignant vs benign)
Size   : 569 instances, 30 numeric features  (>= 500 instances, >= 12 features - meets assignment requirement)
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

# ------------------------------------------------------------------
# 2. Train/test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------------------------
# 3. Scale features (helps Logistic Regression / kNN)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(os.path.join(HERE, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ------------------------------------------------------------------
# 4. Save test data (features + true label) for the Streamlit app
# ------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

# ------------------------------------------------------------------
# 5. Define models
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    # Logistic Regression and kNN benefit from scaled data
    if name in ["Logistic Regression", "kNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    })

    # Save each trained model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    with open(os.path.join(HERE, fname), "wb") as f:
        pickle.dump(model, f)

    print(f"{name:28s} Acc={acc:.4f} AUC={auc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} MCC={mcc:.4f}")

# ------------------------------------------------------------------
# 6. Save comparison table
# ------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(ROOT, "model", "metrics_summary.csv"), index=False)
print("\nSaved metrics_summary.csv, test_data.csv, and all model .pkl files.")
