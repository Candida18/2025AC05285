# Breast Cancer Classification — ML Assignment 2

## a. Problem Statement

Breast cancer diagnosis is a critical medical classification problem where cell-nuclei
measurements taken from a digitized image of a fine needle aspirate (FNA) of a breast
mass are used to predict whether a tumor is **malignant** or **benign**. Early and
accurate classification directly supports timely treatment decisions. This project
implements and compares five supervised classification algorithms — Logistic
Regression, Decision Tree, k-Nearest Neighbors, Naive Bayes, and Random Forest
(Ensemble) — to predict tumor diagnosis, and exposes the trained models through an
interactive Streamlit web application for evaluation.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (also available built-in via
  `sklearn.datasets.load_breast_cancer`, which mirrors the original UCI dataset)
- **Instances:** 569 (≥ 500 required)
- **Features:** 30 numeric features (≥ 12 required) — computed from digitized FNA
  images, describing characteristics of cell nuclei such as radius, texture,
  perimeter, area, smoothness, compactness, concavity, symmetry, and fractal
  dimension (mean, standard error, and "worst"/largest value for each).
- **Target:** Binary classification — `0 = Malignant`, `1 = Benign`
- **Class balance:** 212 malignant, 357 benign
- **Train/Test split:** 80% / 20%, stratified on the target (random_state = 42)

## c. GitHub Repository Link

> **`https://github.com/Candida18/2025AC05285`**

Repository structure:
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
    │-- train_models.py
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest_ensemble.pkl
    │-- scaler.pkl
    │-- metrics_summary.csv
```

## d. Models Used

All 5 models were trained on the **same dataset and same train/test split** described
above. Logistic Regression and kNN were trained on standardised (scaled) features;
Decision Tree, Naive Bayes, and Random Forest were trained on raw features (scale
does not affect them).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

*(Exact figures are reproducible by running `model/train_models.py`; random_state = 42.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset. The classes are close to linearly separable in the standardised feature space, so a simple linear decision boundary generalises very well, giving the highest accuracy, F1, and MCC among all models. |
| Decision Tree | Weakest performer. A single unpruned tree overfits the training data and is sensitive to small variations in the 30 correlated features, which hurts generalisation — it has the lowest accuracy, AUC, and MCC. |
| kNN | Strong performer after feature scaling, since scaling puts all 30 features on comparable ranges for distance computation. Slightly behind Logistic Regression, but ties Random Forest on F1/MCC. |
| Naive Bayes | Decent AUC (probability ranking is good) but lower precision/accuracy than the top models. The Gaussian independence assumption is a simplification given that many of the 30 features (e.g., radius/perimeter/area) are highly correlated, which limits its ceiling. |
| Random Forest (Ensemble) | Very strong and stable performance — bagging many trees fixes the overfitting problem seen in the single Decision Tree, and it achieves the second-highest AUC overall, showing the value of ensembling. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it achieves the highest Accuracy, Precision, Recall, F1, and MCC, with an AUC essentially tied for the best. Random Forest is a close second and is a good ensemble alternative if a non-linear/robust model is preferred. |

## Streamlit App Features

The deployed Streamlit app (`app.py`) provides:
- **CSV upload** of test data (`test_data.csv` provided in this repo)
- **Model selection dropdown** to switch between all 5 trained models
- **Live evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion matrix** heatmap and full **classification report**
- An optional "Run all 5 models and compare" view for side-by-side comparison

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # (optional) retrain models and regenerate test_data.csv
streamlit run app.py
```

## Deployment

Deployed on **Streamlit Community Cloud**:

> **Live App Link:** `https://2025ac05285.streamlit.app/`

Steps used: pushed this repo to GitHub → https://streamlit.io/cloud → signed in with
GitHub → "New App" → selected this repository/branch → set main file to `app.py` →
Deploy.
