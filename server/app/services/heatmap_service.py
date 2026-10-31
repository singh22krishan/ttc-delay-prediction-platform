from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from app.config import HEATMAP_INFERENCE_CONFIG
from app.services.model_service import ModelService

def _ctx_key(vehicle_type: str, month: int, day_of_week: int, hour: int) -> str:
    return f"{vehicle_type}|{month}|{day_of_week}|{hour}"

def load_inference_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Heatmap inference config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {"metadata", "bins", "context_bin_indices"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Outdated heatmap config; missing: {sorted(missing)}")
    return raw["metadata"], raw["bins"], raw["context_bin_indices"]

class HeatmapService:
    def __init__(self, model_service: ModelService) -> None:
        self._model = model_service
        self._metadata, self._bins, self._context_bin_indices = load_inference_config(HEATMAP_INFERENCE_CONFIG)

    def metadata(self) -> dict:
        return {k:self._metadata[k] for k in ("vehicle_types","months","days_of_week","hours")}

    def query(self, *, vehicle_type: str, month: int, day_of_week: int, hour: int, include_time_decay: bool=False) -> dict:
        if vehicle_type not in set(self._metadata["vehicle_types"]):
            raise ValueError(f"Unknown vehicle_type: {vehicle_type}")
        idx = self._context_bin_indices.get(_ctx_key(vehicle_type, month, day_of_week, hour), [])
        if not idx:
            return {"points":[],"kpis":{"point_count":0,"avg_delay":0.0,"p90":0.0}}
        bins = [self._bins[i] for i in idx]
        lats = np.array([b["latitude_bin"] for b in bins], dtype=float)
        lons = np.array([b["longitude_bin"] for b in bins], dtype=float)
        preds = self._model.predict_batch(vehicle_type=vehicle_type, month=month, day_of_week=day_of_week, hour=hour, latitude_bins=lats, longitude_bins=lons, include_time_decay=include_time_decay)
        maximum = max(float(np.max(preds)), 1.0)
        weights = np.clip(preds / maximum, 0.0, 1.0)
        points = [{"latitude_bin":float(lats[i]),"longitude_bin":float(lons[i]),"pred_delay_mean":float(preds[i]),"pred_delay_p90":float(preds[i]),"n_events":1,"weight":float(weights[i])} for i in range(len(preds))]
        return {"points":points,"kpis":{"point_count":len(preds),"avg_delay":float(np.mean(preds)),"p90":float(np.percentile(preds,90))}}
