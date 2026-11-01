# TTC Delay Prediction Platform

A full-stack machine-learning platform for predicting Toronto Transit Commission delay intensity by vehicle type, time context, and geographic location. The project combines a reproducible ML training workflow, FastAPI inference service, and React/TypeScript heatmap client.

## Portfolio Context

Standalone portfolio version maintained by **Krishan Singh**. The implementation originated as an AI Capstone academic project at Centennial College and was shared for course submission. Generated datasets and model binaries are intentionally excluded, but the public repository now includes the reusable training code needed to reproduce the model from an authorized dataset.

## Highlights

- Historical TTC delay modeling using data from 2017–2025
- Temporal train/validation/test workflow designed to reduce leakage
- Feature engineering for month, weekday, hour, weekend, rush hour, peak hour, and night context
- LightGBM and XGBoost model comparison
- Validation-RMSE model selection
- Serialized preprocessing + winning model bundle
- FastAPI REST service with typed Pydantic request/response models
- Vectorized heatmap inference across geographic bins
- Restricted configurable CORS and generic production error responses
- React/TypeScript client for interactive controls, KPIs, and heatmap data
- Pytest coverage for training helpers + GitHub Actions CI

## Architecture

```text
Authorized TTC dataset
        |
        v
aiProject/ttc_pipeline
  |-- feature engineering
  |-- temporal split
  |-- preprocessing
  |-- LightGBM + XGBoost
  |-- validation comparison
  +-- model.pkl export
        |
        v
FastAPI service
  |-- ModelService
  |-- HeatmapService
        |
        v
React + TypeScript client
```

## Reproduce the Training Workflow

The public training module expects a local CSV with at least these fields:

`Date`, `Time`, `Min Delay`, `Vehicle_Type`, `Latitude`, `Longitude`

Place an authorized dataset at:

```text
dataset/ttc_delays_processed.csv
```

Install the Python dependencies and run:

```bash
pip install -r server/requirements.txt
python -m aiProject.ttc_pipeline.cli
```

The training workflow:

1. Parses event date/time and cleans the target.
2. Creates inference-safe temporal and calendar features.
3. Uses 2023-and-earlier for training, 2024 for validation, and 2025 for test when available; otherwise it falls back to chronological 70/15/15 splitting.
4. Caps extreme target values using a training-derived 99.5th percentile threshold.
5. Fits preprocessing only on training data.
6. Trains LightGBM and XGBoost regressors.
7. Selects the best model by validation RMSE.
8. Retrains the selected model on train + validation data.
9. Writes `server/model_artifacts/model.pkl` and `metrics.json` locally.

Model artifacts are ignored by Git and are not committed.

## API

- `GET /health` — service/model status without exposing filesystem paths
- `GET /metadata` — supported filter values
- `GET /heatmap` — batch model inference for a selected context
- `POST /predict` — score a single location/time context

Example prediction payload:

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

## Tests

From the repository root:

```bash
pip install -r server/requirements.txt
pytest -q tests
```

GitHub Actions runs the training-helper tests on pushes and pull requests.

## Security / Privacy

- No model binaries, private datasets, credentials, or local paths are committed.
- CORS origins are supplied through configuration.
- Health checks do not expose artifact paths.
- Unexpected API exceptions return generic messages instead of internal error details.

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · LightGBM · XGBoost · FastAPI · Pydantic · Pytest · React · TypeScript · Vite · GitHub Actions**

## Academic Context

AI Capstone Project — Centennial College, 2026.
