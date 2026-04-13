import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from glob import glob
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
    "Random Forest": "results/loso_summary_rf.json",
    "XGBoost": "results/loso_summary_xgb.json"
}
report_paths = {
    "Random Forest": "results/per_subject_reports_rf.json",
    "XGBoost": "results/per_subject_reports_xgb.json"
}
shap_model_paths = {
    "Random Forest": "results/model_rf.pkl",
    "XGBoost": "results/model_xgb.pkl"
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
report_data = {k: v for k, v in reports[selected_subj].items() if k != 'confusion_matrix'}
report_df = pd.DataFrame(report_data).T.iloc[:-3]  # drop avg rows
report_df = report_df.drop(columns=['support'], errors='ignore').round(3)
col, _ = st.columns([2, 3])
col.dataframe(report_df)

# Confusion Matrix
st.subheader(f"Confusion Matrix – {selected_subj}")
try:
    cm = np.array(reports[selected_subj]['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Baseline", "Stress", "Amusement"],
                yticklabels=["Baseline", "Stress", "Amusement"],
                ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.tick_params(axis='both', labelsize=8)
    col, _ = st.columns([1, 2])
    col.pyplot(fig)
    plt.close(fig)
except KeyError:
    st.warning("Confusion matrix not available for this subject.")

# SHAP summary
if st.checkbox("Show SHAP Explainability (last trained fold)"):
    st.subheader("SHAP Feature Importance")

    @st.cache_resource
    def load_shap(model_path):
        return joblib.load(model_path)

    @st.cache_data
    def load_features():
        files = sorted(glob("data/features/S*_features.csv"))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df[df['label'].isin([1, 2, 3])]
        return df.drop(columns=['label'])

    model = load_shap(shap_model_paths[selected_model])
    X_all = load_features()
    X_sample = X_all.sample(min(200, len(X_all)), random_state=42)

    with st.spinner("Computing SHAP values..."):
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)

    st.caption("Based on last trained fold model — 200 randomly sampled segments.")
    shap.summary_plot(shap_values, features=X_sample, show=False)
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    col, _ = st.columns([2, 3])
    col.pyplot(fig, bbox_inches='tight')
    plt.clf()
