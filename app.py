"""
Streamlit App — Breast Cancer Classification Demo
---------------------------------------------------
Lets a user upload a CSV of test data (matching test_data.csv), pick one of
5 trained models, and see evaluation metrics + confusion matrix / classification
report computed on the uploaded data.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Model Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Load artifacts (cached so they don't reload on every interaction)
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    with open("model/meta.json") as f:
        meta = json.load(f)
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return scaler, meta, models


@st.cache_data
def load_precomputed_metrics():
    return pd.read_csv("model/metrics_comparison.csv")


scaler, meta, models = load_artifacts()
feature_names = meta["feature_names"]
target_names = meta["target_names"]  # ['malignant', 'benign']

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.caption(
    "CSV must contain the same 30 feature columns as the training data, "
    "plus a `target` column (0 = malignant, 1 = benign) for evaluation."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🔬 Breast Cancer Classification — Model Explorer")
st.write(
    "This app demonstrates 5 classification models (Logistic Regression, "
    "Decision Tree, kNN, Naive Bayes, Random Forest) trained on the "
    "Breast Cancer Wisconsin (Diagnostic) dataset."
)

tab1, tab2 = st.tabs(["📊 Evaluate on uploaded data", "📈 Precomputed model comparison"])

# ------------------------- TAB 1: live evaluation -------------------------
with tab1:
    if uploaded_file is None:
        st.info("👈 Upload a CSV (e.g. `test_data.csv` from the repo) to evaluate a model on it.")
    else:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of uploaded data")
        st.dataframe(df.head())

        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        else:
            X = df[feature_names]
            X_scaled = scaler.transform(X)

            model = models[selected_model_name]
            y_pred = model.predict(X_scaled)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_scaled)[:, 1]
            else:
                y_proba = model.decision_function(X_scaled)

            pred_labels = pd.Series(y_pred).map({0: target_names[0], 1: target_names[1]})

            st.subheader(f"Predictions — {selected_model_name}")
            result_view = df.copy()
            result_view["prediction"] = pred_labels.values
            st.dataframe(result_view.head(20))

            if "target" in df.columns:
                y_true = df["target"]

                st.subheader("Evaluation Metrics")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                m2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
                m3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
                m4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
                m5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
                m6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=target_names, yticklabels=target_names, ax=ax
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                with col_b:
                    st.subheader("Classification Report")
                    report = classification_report(
                        y_true, y_pred, target_names=target_names, output_dict=True
                    )
                    st.dataframe(pd.DataFrame(report).transpose().round(3))
            else:
                st.warning(
                    "No `target` column found — showing predictions only. "
                    "Include a `target` column to see metrics, confusion matrix, "
                    "and classification report."
                )

# ------------------------- TAB 2: precomputed comparison -------------------------
with tab2:
    st.subheader("Model comparison (computed during training, on the held-out test split)")
    metrics_df = load_precomputed_metrics()
    st.dataframe(metrics_df.set_index("ML Model Name").style.highlight_max(axis=0, color="lightgreen"))

    st.subheader("Metric comparison chart")
    metric_to_plot = st.selectbox(
        "Choose a metric to visualize", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    )
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.barplot(data=metrics_df, x="ML Model Name", y=metric_to_plot, ax=ax2, palette="viridis")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right")
    ax2.set_ylim(0, 1.05)
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built for BITS Pilani WILP — M.Tech (AIML/DSE) — Machine Learning Assignment 2")
