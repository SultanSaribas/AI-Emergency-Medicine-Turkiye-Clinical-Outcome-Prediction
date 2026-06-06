from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import emergency_ml_pipeline as em

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "xgboost_models.joblib"
METADATA_PATH = ARTIFACT_DIR / "xgboost_metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found at {MODEL_PATH}. "
        "Run the model-saving cell in eda.ipynb first."
    )

models: dict = joblib.load(MODEL_PATH)

with open(METADATA_PATH) as f:
    metadata: dict = json.load(f)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EncounterFeatures(BaseModel):
    # identifiers (optional — returned in response for traceability)
    encounter_id: Optional[str] = Field(default=None)

    # demographics
    age: Optional[float] = None
    age_group: Optional[str] = None
    gender: Optional[str] = None
    is_pediatric: Optional[int] = None
    is_geriatric: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    insurance_type: Optional[str] = None
    ses_index: Optional[float] = None

    # triage & presentation
    arrival_mode: Optional[str] = None
    triage_level: Optional[int] = None
    triage_label: Optional[str] = None
    chief_complaint: Optional[str] = None

    # time & location
    arrival_year: Optional[int] = None
    arrival_month: Optional[int] = None
    arrival_day_of_week: Optional[str] = None
    arrival_hour: Optional[int] = None
    city: Optional[str] = None
    nuts2_region: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    urban_rural: Optional[str] = None
    hospital_type: Optional[str] = None
    annual_ed_volume: Optional[float] = None
    bed_count_category: Optional[str] = None

    # vitals
    heart_rate_bpm: Optional[float] = None
    systolic_bp_mmhg: Optional[float] = None
    diastolic_bp_mmhg: Optional[float] = None
    respiratory_rate_rpm: Optional[float] = None
    temperature_celsius: Optional[float] = None
    spo2_pct: Optional[float] = None
    gcs_total: Optional[float] = None

    # severity scores
    mews_score: Optional[float] = None
    news2_score: Optional[float] = None

    # labs
    wbc_count_mgdl: Optional[float] = None
    hemoglobin_gdl: Optional[float] = None
    platelets_mgdl: Optional[float] = None
    sodium_meql: Optional[float] = None
    potassium_meql: Optional[float] = None
    creatinine_mgdl: Optional[float] = None
    glucose_mgdl: Optional[float] = None
    alt_ul: Optional[float] = None
    ast_ul: Optional[float] = None
    crp_mgdl: Optional[float] = None
    inr: Optional[float] = None
    aptt_seconds: Optional[float] = None
    troponin_i_ngml: Optional[float] = None
    bnp_pgl: Optional[float] = None
    d_dimer_mgfl: Optional[float] = None
    lactate_mmoll: Optional[float] = None
    lipase_ul: Optional[float] = None
    procalcitonin_ngl: Optional[float] = None
    albumin_gdl: Optional[float] = None
    total_bilirubin_mgdl: Optional[float] = None
    egfr_ml_min: Optional[float] = None

    # comorbidities
    comorbidity_hypertension: Optional[int] = None
    comorbidity_diabetes: Optional[int] = None
    comorbidity_chf: Optional[int] = None
    comorbidity_copd: Optional[int] = None
    comorbidity_ckd: Optional[int] = None
    comorbidity_cancer: Optional[int] = None
    comorbidity_stroke: Optional[int] = None
    comorbidity_afib: Optional[int] = None
    comorbidity_liver_disease: Optional[int] = None
    comorbidity_obesity: Optional[int] = None
    polypharmacy_5plus: Optional[int] = None

    # procedures & consult
    iv_fluids_ml: Optional[float] = None
    consult_requested: Optional[int] = None
    consult_specialty: Optional[str] = None

    # diagnosis
    primary_icd10_code: Optional[str] = None
    primary_icd10_category: Optional[str] = None
    pandemic_period: Optional[str] = None
    covid_suspected: Optional[int] = None
    covid_confirmed: Optional[int] = None

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    encounter_id: Optional[str]
    adverse_outcome_prob: float
    readmission_30d_prob: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Emergency Medicine Outcome Prediction API",
    version="1.0.0",
)


def _predict_single(features: dict) -> dict:
    row = pd.DataFrame([features])
    encounter_id = row.pop("encounter_id").iloc[0] if "encounter_id" in row.columns else None

    X = em.make_feature_matrix(row)

    # align columns to the training schema — fill any missing columns with NaN
    first_pipeline = next(iter(models.values()))
    preprocessor = first_pipeline.named_steps["preprocess"]
    expected_cols = (
        list(preprocessor.transformers_[0][2]) +
        list(preprocessor.transformers_[1][2])
    )
    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan

    probs = {}
    for target, pipeline in models.items():
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(X)[0, 1]
        else:
            score = pipeline.decision_function(X)[0]
            prob = float(1 / (1 + np.exp(-score)))
        probs[target] = round(float(prob), 6)

    return {
        "encounter_id": encounter_id,
        "adverse_outcome_prob": probs.get("adverse_outcome", 0.0),
        "readmission_30d_prob": probs.get("readmission_30d", 0.0),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": metadata.get("model_type"),
        "targets": metadata.get("targets"),
        "created_at": metadata.get("created_at"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: EncounterFeatures):
    try:
        result = _predict_single(encounter.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(encounters: list[EncounterFeatures]):
    if not encounters:
        raise HTTPException(status_code=400, detail="Empty request body.")
    try:
        predictions = [_predict_single(e.model_dump()) for e in encounters]
        return {"predictions": predictions}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
