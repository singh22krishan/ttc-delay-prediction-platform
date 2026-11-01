from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.schemas.types import PredictRequest, PredictResponse
from app.services.heatmap_service import HeatmapService
from app.services.model_service import ModelService

app = FastAPI(title="TTC Delay Heatmap API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
model_service = ModelService()
heatmap_service = HeatmapService(model_service)


@app.get("/health")
def health():
    return {"status": "ok", "model": model_service.model_name}


@app.get("/metadata")
def metadata():
    return heatmap_service.metadata()


@app.get("/heatmap")
def heatmap(
    vehicle_type: str = Query(...),
    month: int = Query(..., ge=1, le=12),
    day_of_week: int = Query(..., ge=0, le=6),
    hour: int = Query(..., ge=0, le=23),
    include_time_decay: bool = Query(False),
):
    try:
        return heatmap_service.query(
            vehicle_type=vehicle_type,
            month=month,
            day_of_week=day_of_week,
            hour=hour,
            include_time_decay=include_time_decay,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to generate heatmap") from exc


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        prediction = model_service.predict_single(
            vehicle_type=payload.vehicle_type,
            month=payload.month,
            day_of_week=payload.day_of_week,
            hour=payload.hour,
            latitude=payload.latitude,
            longitude=payload.longitude,
            include_time_decay=payload.include_time_decay,
        )
        return PredictResponse(
            predicted_delay_minutes=prediction,
            model_name=model_service.model_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to score prediction") from exc
