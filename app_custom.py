# ===========================================================
# Smart Manufacturing Output Estimator
# FastAPI Version 3.1 (Enhanced for Production Use)
# ===========================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os
from datetime import datetime

# ===========================================================
# FastAPI App Initialization
# ===========================================================
app = FastAPI(
    title="Smart Manufacturing Output Estimator",
    description="A Linear Regression-based API to predict hourly machine output \
                using manufacturing parameters and operational metrics.",
    version="3.1"
)

# ===========================================================
# Load Model and Scaler
# ===========================================================
try:
    model = pickle.load(open("models/output_predictor.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
except Exception as e:
    raise RuntimeError("Error loading model or scaler: " + str(e))

# ===========================================================
# Input Schema
# ===========================================================
class MachineInput(BaseModel):
    Mold_Temp: float
    Line_Pressure: float
    Cycle_Duration: float
    Cool_Period: float
    Viscosity_Index: float
    Env_Temp: float
    Machine_Years: float
    Operator_Skill: float
    Maintenance_Time: float


# ===========================================================
# Root Endpoint
# ===========================================================
@app.get("/")
def home():
    return {
        "Welcome": "Smart Manufacturing Output Estimator API",
        "Docs": "Visit /docs for Swagger UI 🧠",
        "Tips": "Ensure your numeric inputs are realistic values matching machine configurations."
    }


# ===========================================================
# Prediction Endpoint
# ===========================================================
@app.post("/predict")
def predict_output(data: MachineInput):
    try:
        # 1️⃣ Derived features (match training logic)
        ratio = data.Line_Pressure / data.Mold_Temp
        efficiency = (data.Operator_Skill / (data.Cycle_Duration + data.Cool_Period)) * 100
        cooling_eff = data.Cool_Period / data.Cycle_Duration

        # 2️⃣ Prepare feature array
        features = np.array([[
            data.Mold_Temp, data.Line_Pressure, data.Cycle_Duration, data.Cool_Period,
            data.Viscosity_Index, data.Env_Temp, data.Machine_Years,
            data.Operator_Skill, data.Maintenance_Time,
            ratio, efficiency, cooling_eff
        ]])

        # 3️⃣ Scale data
        scaled_features = scaler.transform(features)

        # 4️⃣ Predict
        prediction = model.predict(scaled_features)[0]
        status = "Optimal ✅" if prediction > 300 else "Below Average ⚠️"
        suggestion = (
            "Maintain stable mold temperature and consistent operator performance."
            if prediction > 300 else
            "Increase pressure and reduce cooling time to improve output."
        )

        # 5️⃣ Save log
        os.makedirs("logs", exist_ok=True)
        with open("logs/api_requests_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | Predicted={prediction:.2f} | Status={status}\n")

        # 6️⃣ Return structured JSON
        return {
            "Predicted Parts/Hour": round(prediction, 2),
            "Performance Status": status,
            "Insight": suggestion,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ===========================================================
# Monitoring Endpoint (Health Check)
# ===========================================================
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": True,
        "scaler_loaded": True,
        "last_updated": "2025-10-26",
        "message": "API is healthy and ready to serve predictions 🚀"
    }
