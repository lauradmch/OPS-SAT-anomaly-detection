import os

import mlflow
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

# API's model URI env-driven -> code loads the alias locally or a local path in the container
MODEL_URI = os.getenv("MODEL_URI", "models:/ops-sat-anomaly-detector@production")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = mlflow.pyfunc.load_model(MODEL_URI)   # load ONCE at startup
    yield
    state.clear()


app = FastAPI(title="OPS-SAT Anomaly Detector", version="1.0", lifespan=lifespan)


class SegmentRequest(BaseModel):
    channel: str
    values: list[float] = Field(min_length=2)             # reject degenerate segments


class PredictionResponse(BaseModel):
    channel: str
    score: float
    is_anomaly: bool
    threshold: float
    ae_pct: float
    hp_pct: float
    n_points: int
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI, "loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: SegmentRequest):
    df = pd.DataFrame([req.model_dump()])
    return state["model"].predict(df)[0]