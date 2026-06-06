from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"


@dataclass
class PatientData:
    age: Optional[float] = None
    age_group: Optional[str] = None
    gender: Optional[str] = None
    is_pediatric: Optional[int] = None
    is_geriatric: Optional[int] = None
    triage_level: Optional[int] = None
    chief_complaint: Optional[str] = None
    arrival_mode: Optional[str] = None
    heart_rate_bpm: Optional[float] = None
    systolic_bp_mmhg: Optional[float] = None
    diastolic_bp_mmhg: Optional[float] = None
    respiratory_rate_rpm: Optional[float] = None
    temperature_celsius: Optional[float] = None
    spo2_pct: Optional[float] = None
    gcs_total: Optional[float] = None
    mews_score: Optional[float] = None
    news2_score: Optional[float] = None
    wbc_count_mgdl: Optional[float] = None
    hemoglobin_gdl: Optional[float] = None
    creatinine_mgdl: Optional[float] = None
    glucose_mgdl: Optional[float] = None
    troponin_i_ngml: Optional[float] = None
    lactate_mmoll: Optional[float] = None
    comorbidity_hypertension: Optional[int] = None
    comorbidity_diabetes: Optional[int] = None
    comorbidity_chf: Optional[int] = None
    comorbidity_copd: Optional[int] = None
    comorbidity_ckd: Optional[int] = None


class APIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def predict(self, patient: PatientData) -> dict:
        payload = {k: v for k, v in asdict(patient).items() if v is not None}
        r = requests.post(f"{self.base_url}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()


class EmergencyPredictionApp:
    def __init__(self, client: APIClient):
        self.client = client

    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.header("API Status")
            try:
                info = self.client.health()
                st.success("Connected")
                st.caption(f"Model: {info.get('model_type', '-')}")
                st.caption(f"Created: {str(info.get('created_at', ''))[:10]}")
            except Exception:
                st.error("API unreachable")
                st.caption(f"Expected at: {self.client.base_url}")

    def _build_patient(self) -> PatientData:
        t_demo, t_triage, t_vitals, t_labs, t_comorbid = st.tabs(
            ["Demographics", "Triage", "Vitals", "Labs", "Comorbidities"]
        )

        with t_demo:
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", min_value=0, max_value=120, value=None)
            gender = c2.selectbox(
                "Gender",
                [None, "M", "F", "Other"],
                format_func=lambda x: "—" if x is None else x,
            )
            age_group = c1.selectbox(
                "Age Group",
                [None, "pediatric", "adult", "geriatric"],
                format_func=lambda x: "—" if x is None else x,
            )

        with t_triage:
            c1, c2 = st.columns(2)
            triage_level = c1.selectbox(
                "Triage Level",
                [None, 1, 2, 3, 4, 5],
                format_func=lambda x: "—" if x is None else str(x),
            )
            arrival_mode = c2.selectbox(
                "Arrival Mode",
                [None, "ambulance", "walk-in", "helicopter", "transfer"],
                format_func=lambda x: "—" if x is None else x,
            )
            chief_complaint = st.text_input("Chief Complaint")

        with t_vitals:
            c1, c2, c3 = st.columns(3)
            hr = c1.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=None)
            sbp = c2.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=None)
            dbp = c3.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=None)
            rr = c1.number_input("Resp. Rate (rpm)", min_value=0, max_value=60, value=None)
            temp = c2.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=None)
            spo2 = c3.number_input("SpO2 (%)", min_value=0, max_value=100, value=None)
            gcs = c1.number_input("GCS Total", min_value=3, max_value=15, value=None)
            mews = c2.number_input("MEWS Score", min_value=0, max_value=20, value=None)
            news2 = c3.number_input("NEWS2 Score", min_value=0, max_value=20, value=None)

        with t_labs:
            c1, c2, c3 = st.columns(3)
            wbc = c1.number_input("WBC (mg/dL)", min_value=0.0, value=None)
            hgb = c2.number_input("Hemoglobin (g/dL)", min_value=0.0, value=None)
            creatinine = c3.number_input("Creatinine (mg/dL)", min_value=0.0, value=None)
            glucose = c1.number_input("Glucose (mg/dL)", min_value=0.0, value=None)
            troponin = c2.number_input("Troponin I (ng/mL)", min_value=0.0, value=None)
            lactate = c3.number_input("Lactate (mmol/L)", min_value=0.0, value=None)

        with t_comorbid:
            c1, c2 = st.columns(2)
            hypertension = c1.checkbox("Hypertension")
            diabetes = c2.checkbox("Diabetes")
            chf = c1.checkbox("CHF")
            copd = c2.checkbox("COPD")
            ckd = c1.checkbox("CKD")

        return PatientData(
            age=float(age) if age is not None else None,
            age_group=age_group,
            gender=gender,
            is_pediatric=1 if age_group == "pediatric" else (0 if age_group else None),
            is_geriatric=1 if age_group == "geriatric" else (0 if age_group else None),
            triage_level=triage_level,
            chief_complaint=chief_complaint or None,
            arrival_mode=arrival_mode,
            heart_rate_bpm=float(hr) if hr is not None else None,
            systolic_bp_mmhg=float(sbp) if sbp is not None else None,
            diastolic_bp_mmhg=float(dbp) if dbp is not None else None,
            respiratory_rate_rpm=float(rr) if rr is not None else None,
            temperature_celsius=float(temp) if temp is not None else None,
            spo2_pct=float(spo2) if spo2 is not None else None,
            gcs_total=float(gcs) if gcs is not None else None,
            mews_score=float(mews) if mews is not None else None,
            news2_score=float(news2) if news2 is not None else None,
            wbc_count_mgdl=float(wbc) if wbc is not None else None,
            hemoglobin_gdl=float(hgb) if hgb is not None else None,
            creatinine_mgdl=float(creatinine) if creatinine is not None else None,
            glucose_mgdl=float(glucose) if glucose is not None else None,
            troponin_i_ngml=float(troponin) if troponin is not None else None,
            lactate_mmoll=float(lactate) if lactate is not None else None,
            comorbidity_hypertension=int(hypertension),
            comorbidity_diabetes=int(diabetes),
            comorbidity_chf=int(chf),
            comorbidity_copd=int(copd),
            comorbidity_ckd=int(ckd),
        )

    @staticmethod
    def _risk_level(prob: float) -> tuple[str, str]:
        if prob < 0.3:
            return "Low", "success"
        if prob < 0.6:
            return "Moderate", "warning"
        return "High", "error"

    def _show_results(self, result: dict) -> None:
        st.divider()
        st.subheader("Prediction Results")
        c1, c2 = st.columns(2)
        for col, key, label in [
            (c1, "adverse_outcome_prob", "Adverse Outcome"),
            (c2, "readmission_30d_prob", "30-Day Readmission"),
        ]:
            prob = result[key]
            risk, status = self._risk_level(prob)
            col.metric(label, f"{prob:.1%}")
            getattr(col, status)(f"Risk: {risk}")

    def run(self) -> None:
        st.set_page_config(
            page_title="Emergency Outcome Prediction",
            layout="wide",
        )
        st.title("Emergency Medicine Clinical Outcome Prediction")
        self._render_sidebar()

        with st.form("patient_form"):
            patient = self._build_patient()
            submitted = st.form_submit_button(
                "Predict", type="primary", use_container_width=True
            )

        if submitted:
            with st.spinner("Running prediction..."):
                try:
                    result = self.client.predict(patient)
                    self._show_results(result)
                except requests.ConnectionError:
                    st.error(
                        "Cannot connect to API. "
                        "Start it with: uvicorn app:app --reload"
                    )
                except requests.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")


if __name__ == "__main__":
    EmergencyPredictionApp(APIClient()).run()
