# TTC Delay Prediction Platform

A full-stack machine-learning platform for predicting Toronto Transit Commission delay intensity by vehicle type, time context, and geographic location. The project combines a reusable ML training workflow, FastAPI inference service, and React/TypeScript heatmap client.

## Portfolio Context

Standalone portfolio version maintained by **Krishan Singh**. The implementation originated as an AI Capstone academic project at Centennial College and was shared for course submission. This version focuses on the application architecture and maintainable source code rather than generated datasets, caches, or model binaries.

## Highlights

- Historical TTC delay modeling using data from 2017–2025
- Temporal train/validation/test workflow designed to reduce leakage
- Feature engineering for month, weekday, hour, weekend, rush hour, peak hour, and night context
- LightGBM and XGBoost model comparison
- Serialized preprocessing + winning model bundle
- FastAPI REST service with typed Pydantic request/response models
- Vectorized heatmap inference across geographic bins
- Context-aware heatmap filtering by vehicle, month, weekday, and hour
- React/TypeScript client for interactive controls, KPIs, and heatmap data

## Architecture

```text
React + TypeScript client
        |
        | GET /metadata
        | GET /heatmap
        | POST /predict
        v
FastAPI service
  |-- HeatmapService
  |-- ModelService
        |
        +-- model.pkl
        +-- heatmap_inference_config.json

ML training pipeline
  |-- preprocessing
  |-- LightGBM
  |-- XGBoost
  |-- model comparison
  +-- production artifact export
```

## API

### `GET /health`
Reports API health and configured artifact paths.

### `GET /metadata`
Returns supported vehicle types and time-domain filter values.

### `GET /heatmap`
Runs batch model inference for the selected context and returns map points plus KPI aggregates.

### `POST /predict`
Scores a single location/time context and returns predicted delay minutes and model name.

Example payload:

```json
{
  "vehicle_type": "BUS",
  "month": 10,
  "day_of_week": 2,
  "hour": 17,
  "latitude": 43.6532,
  "longitude": -79.3832,
  "include_time_decay": false
}
```

## ML Design

The original capstone workflow engineered inference-safe features and compared LightGBM with XGBoost on temporal validation/test data. The production artifact contains both the preprocessing transformer and selected regressor so API inference uses the same feature transformation as training.

The portfolio version intentionally does **not** commit generated datasets or `model.pkl`. Run the training pipeline against the project dataset to generate artifacts locally.

## Run the API

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run the Client

```bash
cd client
npm install
npm run dev
```

Set `VITE_API_URL` if the API is not running on `http://localhost:8000`.

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · LightGBM · XGBoost · FastAPI · Pydantic · REST APIs · React · TypeScript · Vite**

## Academic Context

AI Capstone Project — Centennial College, 2026.
