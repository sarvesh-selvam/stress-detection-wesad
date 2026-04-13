# Stress Detection – WESAD

Physiological stress detection using the [WESAD dataset](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection). Classifies three states — **Baseline**, **Stress**, and **Amusement** — from chest-worn sensor signals using classical machine learning with subject-independent evaluation.

---

## Dataset

**WESAD** (Wearable Stress and Affect Detection) contains multimodal physiological recordings from 15 subjects (S2–S17, excluding S12) wearing a RespiBAN chest device and an Empatica E4 wrist device.

This project uses chest signals only:

| Signal | Sensor | Sampling Rate | Filter Applied |
|---|---|---|---|
| ECG | RespiBAN | 700 Hz | Bandpass 0.5–40 Hz |
| EDA | RespiBAN | 700 Hz | Lowpass 1 Hz |
| Respiration | RespiBAN | 700 Hz | Lowpass 1 Hz |

**Labels used:** `1` = Baseline, `2` = Stress, `3` = Amusement (other protocol labels excluded)

---

## Pipeline

### 1. Preprocessing (`01_exploratory_analysis.ipynb`)
- Bandpass filters ECG (0.5–40 Hz); lowpass filters EDA and respiration (1 Hz)
- Z-score normalizes each signal
- Segments into **60-second windows with 50% overlap** (42,000 samples per window at 700 Hz)
- Assigns labels via majority vote within each window
- Saves segments and labels as `.npy` files per subject

### 2. Feature Engineering (`02_feature_engineering.ipynb`)
Extracts 12 hand-crafted physiological features per segment:

| Domain | Features |
|---|---|
| HRV (ECG) | `mean_rr`, `sdnn`, `rmssd`, `ecg_peak_count` |
| Electrodermal (EDA) | `eda_mean`, `eda_std`, `eda_peak_count`, `eda_slope` |
| Respiration | `resp_mean`, `resp_std`, `resp_zero_crossings` |

R-peaks are detected using a 75th-percentile height threshold and a minimum 0.5s inter-peak distance.

### 3. Modeling (`03_modeling.ipynb`)
- **Evaluation:** Leave-One-Subject-Out (LOSO) cross-validation — trains on 14 subjects, tests on the held-out subject, repeated for all 15
- **Hyperparameter tuning:** `RandomizedSearchCV` (20 iterations, 3-fold CV) run inside each LOSO fold on training subjects only
- **Models:** Random Forest and XGBoost

---

## Results

| Model | Mean Accuracy | Mean F1 (weighted) |
|---|---|---|
| Random Forest | 82.9% | 80.7% |
| XGBoost | 81.4% | ~79% |

Top features by importance: `eda_mean`, `eda_peak_count`, `resp_std`, `mean_rr`

---

## Dashboard

An interactive Streamlit dashboard for exploring results:

```bash
streamlit run streamlit-app.py
```

Features:
- Model selector (Random Forest / XGBoost)
- Per-subject classification report
- Confusion matrix heatmap
- SHAP feature importance

---

## Setup

```bash
pip install -r requirements.txt
```

**Run notebooks in order:**
1. `notebooks/01_exploratory_analysis.ipynb` — preprocessing
2. `notebooks/02_feature_engineering.ipynb` — feature extraction
3. `notebooks/03_modeling.ipynb` — training and evaluation

**Note:** Raw WESAD data must be downloaded separately and placed under `data/raw/S{id}/S{id}.pkl`. The dataset is available from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).

---

## Project Structure

```
stress-detection-wesad/
├── data/
│   ├── raw/              # Original WESAD .pkl files (not included)
│   ├── processed/        # Segmented signals as .npy files
│   └── features/         # Extracted feature CSVs per subject
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── results/              # Saved models, LOSO summaries, per-subject reports
├── streamlit-app.py
└── requirements.txt
```
