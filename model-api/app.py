import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# Fixed model path inside container; ensure your Compose mounts ./models to /models
MODEL_PATH = "/models/best_pipeline.joblib"

app = FastAPI(title="Trading Model API", version="1.0.0")

class PredictRequest(BaseModel):
    year: int
    month: int
    day_of_week: int
    open: float
    volume: float
    return_prev_close: float
    volatility_n_days: float
    is_monday: int
    is_friday: int

class PredictResponse(BaseModel):
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
        label = int(model.predict(df)[0])
        return PredictResponse(pred_label=label)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))