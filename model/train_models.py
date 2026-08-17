"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluates them with Accuracy, AUC, Precision, Recall, F1, and MCC, and saves:
  - trained models (joblib)
  - the fitted scaler
  - test_data.csv (used later by the Streamlit app)
  - metrics_comparison.csv (used to fill the README table)

Dataset: sklearn.datasets.load_breast_cancer
  - 569 instances, 30 numeric features, binary target (malignant / benign)
  - satisfies assignment minimums (>=12 features, >=500 instances)
"""

import json
import joblib
import numpy as np
import pandas as pd
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

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.frame[data.feature_names]
y = data.frame["target"]  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}, classes: {sorted(y.unique())}")

# ---------------------------------------------------------------------------
# 2. Train/test split (stratified, 80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Scale features (fit on train only, then transform both)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# ---------------------------------------------------------------------------
# 4. Save test data (features + true label) — this is the CSV the Streamlit
#    app will let a user upload, and what gets committed to the repo.
# ---------------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with shape {test_df.shape}")

# ---------------------------------------------------------------------------
# 5. Define models
#    Tree/ensemble/NB models don't strictly need scaling, but using the
#    scaled features consistently keeps the pipeline simple and uniform
#    across all 5 models (and matches what the Streamlit app will do).
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    # AUC needs probability of the positive class
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # Save the model
    fname = "model/" + name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    joblib.dump(model, fname)

# ---------------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model/metrics_comparison.csv", index=False)
print("\nFinal comparison table:")
print(results_df.to_string(index=False))

# Save feature names + target names for the Streamlit app
meta = {
    "feature_names": list(data.feature_names),
    "target_names": list(data.target_names),  # ['malignant', 'benign']
}
with open("model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nDone. Models, scaler, metadata, and metrics saved in model/")
