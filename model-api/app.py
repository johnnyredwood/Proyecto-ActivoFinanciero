import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "/home/jovyan/work/models/best_pipeline.joblib")

app = FastAPI(title="Trading Model API", version="1.0.0")

class PredictRequest(BaseModel):
    # Minimal feature schema; expand as needed
    year: int
    month: int
    day_of_week: int
    open: float
    high: float
    low: float
    close_prev: float
    adj_close: float
    volume: float
    return_prev_close: float
    volatility_n_days: float
    is_monday: int
    is_friday: int
    ticker: str

class PredictResponse(BaseModel):
    prob_up: float | None
    pred_label: int

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([req.dict()])
        if hasattr(model.named_steps.get('clf'), 'predict_proba'):
            proba = float(model.predict_proba(df)[:, 1][0])
            label = int(proba >= float(os.getenv('PRED_THRESHOLD', '0.5')))
            return PredictResponse(prob_up=proba, pred_label=label)
        else:
            label = int(model.predict(df)[0])
            return PredictResponse(prob_up=None, pred_label=label)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
