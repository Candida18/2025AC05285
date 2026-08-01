"""
app.py — Streamlit Web App for Breast Cancer Classification
Assignment 2 — Machine Learning (BITS WILP)

Features:
  a. CSV upload for test data
  b. Model selection dropdown
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")
st.title("🔬 Breast Cancer Classification — Model Comparison App")
st.caption("2025AC05285 | ML Assignment 2 — Logistic Regression | Decision Tree | kNN | Naive Bayes | Random Forest")

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

SCALED_MODELS = {"Logistic Regression", "kNN"}

# ------------------------------------------------------------------
# Cache model / scaler loading
# ------------------------------------------------------------------
@st.cache_resource
def load_model(filename):
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        return pickle.load(f)

# ------------------------------------------------------------------
# Sidebar — dataset upload + model selection
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV) — must include a 'target' column",
    type=["csv"]
)

model_choice = st.sidebar.selectbox("Select a Model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload `test_data.csv` (provided in the repo) or your own test split with the "
    "same 30 feature columns from the Breast Cancer Wisconsin dataset plus a `target` column."
)

# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column with the true labels.")
        st.stop()

    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    st.subheader("📄 Preview of Uploaded Test Data")
    st.dataframe(df.head())

    model = load_model(MODEL_FILES[model_choice])

    if model_choice in SCALED_MODELS:
        scaler = load_scaler()
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]

    # ---------------- Metrics ----------------
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader(f"📊 Evaluation Metrics — {model_choice}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # ---------------- Confusion Matrix ----------------
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------- Classification Report ----------------
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4))

    # ---------------- Compare all models ----------------
    st.subheader("🏆 Compare All Models on This Test Data")
    if st.checkbox("Run all 5 models and compare"):
        rows = []
        scaler = load_scaler()
        for name, fname in MODEL_FILES.items():
            m = load_model(fname)
            X_in = scaler.transform(X_test) if name in SCALED_MODELS else X_test
            yp = m.predict(X_in)
            ypr = m.predict_proba(X_in)[:, 1]
            rows.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, yp), 4),
                "AUC": round(roc_auc_score(y_test, ypr), 4),
                "Precision": round(precision_score(y_test, yp), 4),
                "Recall": round(recall_score(y_test, yp), 4),
                "F1": round(f1_score(y_test, yp), 4),
                "MCC": round(matthews_corrcoef(y_test, yp), 4),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"))

else:
    st.info("👈 Upload a CSV test file from the sidebar to get started. "
            "You can use the `test_data.csv` provided in this repository.")
    st.markdown("""
    ### About this app
    This app demonstrates 5 classification models trained on the
    **Breast Cancer Wisconsin (Diagnostic)** dataset (569 instances, 30 features):

    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors
    4. Gaussian Naive Bayes
    5. Random Forest (Ensemble)

    Upload the provided `test_data.csv` and pick a model from the sidebar to see its
    performance metrics, confusion matrix, and classification report.
    """)
