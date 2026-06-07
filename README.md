# AI Emergency Medicine — Türkiye Clinical Outcome Prediction

Predicting adverse clinical outcomes and 30-day readmission risk from a large-scale emergency department dataset from Türkiye.

**Live demo:** [streamlit.app](https://ai-emergency-medicine-turkiye-clinical-outcome-prediction.streamlit.app/)

---

## Overview

This project tackles two binary classification tasks on emergency department encounter data:

- **Adverse Outcome** — whether the patient experiences a serious adverse event during the visit
- **30-Day Readmission** — whether the patient is readmitted within 30 days of discharge

Multiple models (SGD, Random Forest, XGBoost, LightGBM) were trained and compared. XGBoost was selected for deployment based on validation ROC-AUC.

---

## Project Structure

```
├── data/                        # Raw dataset (not tracked in git)
├── artifacts/                   # Saved model artifacts (joblib)
├── emergency_ml_pipeline.py     # Feature engineering and model pipeline
├── app.py                       # FastAPI REST API
├── streamlit_app.py             # Streamlit web interface
├── eda.ipynb                    # Exploratory data analysis and model comparison
└── requirements.txt
```

---

## Pipeline

1. Missing value imputation (median for numeric, most frequent for categorical)
2. Informative missingness indicators for key lab values
3. One-hot encoding with minimum frequency threshold
4. StandardScaler for numeric features
5. Classifier (configurable: SGD / Random Forest / XGBoost / LightGBM)

---

## Results

Validation split: 80/20 stratified, ROC-AUC metric.

| Model | Adverse Outcome | 30-Day Readmission | Mean |
|---|---|---|---|
| **XGBoost** | **0.80450** | **0.59904** | **0.70177** |
| LightGBM | 0.80443 | 0.59867 | 0.70155 |
| Random Forest | 0.79769 | 0.59551 | 0.69660 |
| SGD | 0.79004 | 0.57956 | 0.68480 |

XGBoost was selected for deployment based on highest mean ROC-AUC.

---

## Running Locally

```bash
pip install -r requirements.txt

# FastAPI
uvicorn app:app --reload

# Streamlit
streamlit run streamlit_app.py
```

API docs available at `http://127.0.0.1:8000/docs`

---

## Dataset

The dataset is from a Kaggle competition on emergency medicine clinical outcomes in Türkiye. It is not included in this repository.
