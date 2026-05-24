from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# =========================
# LOAD MODEL
# =========================
model = joblib.load("app/models/burnout_model.joblib")
scaler = joblib.load("app/models/scaler.joblib")
features = joblib.load("app/models/features.joblib")

# =========================
# LABEL
# =========================
risk_labels = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Tenang.in API is running"
    }

# =========================
# REQUEST BODY
# =========================
class BurnoutInput(BaseModel):
    Jam_Tidur_Semalam: float
    Seberapa_Sibuk_Anda_Hari_Ini: float
    Suasana_Hati_Anda: float
    prob_anger: float
    prob_anticipation: float
    prob_disgust: float
    prob_fear: float
    prob_joy: float
    prob_sadness: float
    prob_trust: float

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(data: BurnoutInput):

    input_dict = {
        'Jam Tidur Semalam': data.Jam_Tidur_Semalam,
        'Seberapa Sibuk Anda Hari Ini (1-5)': data.Seberapa_Sibuk_Anda_Hari_Ini,
        'Suasana Hati Anda (1-5)': data.Suasana_Hati_Anda,
        'prob_anger': data.prob_anger,
        'prob_anticipation': data.prob_anticipation,
        'prob_disgust': data.prob_disgust,
        'prob_fear': data.prob_fear,
        'prob_joy': data.prob_joy,
        'prob_sadness': data.prob_sadness,
        'prob_trust': data.prob_trust,
    }

    # dataframe
    df = pd.DataFrame([input_dict])

    # urutan feature harus sama
    df = df[features]

    # scaling
    scaled = scaler.transform(df)

    # prediction
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    return {
        "prediction": int(pred),
        "risk_level": risk_labels[int(pred)],
        "probabilities": {
            "Low Risk": float(prob[0]),
            "Medium Risk": float(prob[1]),
            "High Risk": float(prob[2]),
        }
    }
