# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import shap
import joblib

st.set_page_config(page_title="WESAD Stress Detection Dashboard", layout="wide")
st.title("📊 WESAD Stress Detection – LOSO Evaluation Dashboard")

# Model options
model_options = ["Random Forest", "XGBoost"]
selected_model = st.sidebar.selectbox("Select Model", model_options)

# File paths per model
summary_paths = {
    "Random Forest": "loso_summary_rf.json",
    "XGBoost": "loso_summary_xgb.json"
}
report_paths = {
    "Random Forest": "per_subject_reports_rf.json",
    "XGBoost": "per_subject_reports_xgb.json"
}
shap_model_paths = {
    "Random Forest": "model_rf.pkl",
    "XGBoost": "model_xgb.pkl"
}

# Load summary data
@st.cache_data
def load_data(summary_path, reports_path):
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    with open(reports_path, 'r') as f:
        reports = json.load(f)
    return summary, reports

summary, reports = load_data(summary_paths[selected_model], report_paths[selected_model])

# Sidebar subject selector
subject_ids = list(reports.keys())
selected_subj = st.sidebar.selectbox("Select Subject", subject_ids)

# LOSO Summary
st.subheader("LOSO Summary")
col1, col2 = st.columns(2)
col1.metric("Average Accuracy", f"{summary['mean_accuracy']:.2%}")
col2.metric("Average F1 Score", f"{summary['mean_f1']:.2%}")

# Per-subject classification report
st.subheader(f"Classification Report – {selected_subj} ({selected_model})")
report_df = pd.DataFrame(reports[selected_subj]).T.iloc[:-3]  # drop avg rows
report_df = report_df.round(3)
st.dataframe(report_df, use_container_width=True)

# Confusion Matrix
st.subheader(f"Confusion Matrix – {selected_subj}")
try:
    cm = np.array(reports[selected_subj]['confusion_matrix'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Baseline", "Stress", "Amusement"],
                yticklabels=["Baseline", "Stress", "Amusement"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
except KeyError:
    st.warning("Confusion matrix not available for this subject.")

# SHAP summary (optional)
if st.checkbox("Show SHAP Explainability (last model only)"):
    st.subheader("SHAP Feature Importance")
    model = joblib.load(shap_model_paths[selected_model])
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(model.get_booster().get_dump())
    st.write("Feature importance based on last trained model:")
    shap.summary_plot(shap_values, features=model.feature_names_in_, show=False)
    st.pyplot(bbox_inches='tight')
