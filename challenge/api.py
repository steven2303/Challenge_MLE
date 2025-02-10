from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import fastapi
import uvicorn
from pydantic import BaseModel, validator
from typing import List, Dict
import pandas as pd
from datetime import datetime
import numpy as np
from challenge.model import DelayModel
import joblib
from pathlib import Path

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = DelayModel()

categories = joblib.load("models/categories.joblib")
VALID_OPERATORS = categories['valid_airlines']

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator('MES')
    def validate_mes(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('MES must be between 1 and 12')
        return v

    @validator('TIPOVUELO')
    def validate_tipovuelo(cls, v):
        if v not in ['N', 'I']:
            raise ValueError('TIPOVUELO must be either N or I')
        return v
    
    @validator('OPERA')
    def validate_opera(cls, v):
        if v not in VALID_OPERATORS:
            raise ValueError(f'Invalid operator: {v}. Valid operators are: {", ".join(sorted(VALID_OPERATORS))}')
        return v

class PredictRequest(BaseModel):
    flights: List[Flight]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    try:
        data = pd.DataFrame([flight.dict() for flight in request.flights])
        features = model.preprocess(data)
        predictions = model.predict(features)
        return {
            "predict": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)