from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from app.config import MODEL_FILE

class ModelService:
    def __init__(self) -> None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        payload = joblib.load(MODEL_FILE)
        self.model = payload["model"]
        self.preprocessor = payload["preprocessor"]
        self.features = payload["features"]
        self.model_name = payload.get("best_model_name", "unknown")

    @staticmethod
    def _time_flags(day_of_week: int, hour: int) -> dict:
        return {
            "IsWeekend": int(day_of_week in (5, 6)),
            "IsRushHour": int(hour in (7, 8, 9, 16, 17, 18)),
            "IsPeakHour": int(hour in (6, 7, 8, 9, 15, 16, 17, 18, 19)),
            "IsNight": int(hour in (0, 1, 2, 3, 4, 5, 23)),
        }

    def _to_feature_frame(self, transformed: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(transformed, columns=self.preprocessor.get_feature_names_out())

    def predict_single(self, *, vehicle_type: str, month: int, day_of_week: int, hour: int, latitude: float, longitude: float, include_time_decay: bool = False) -> float:
        row = {"Vehicle_Type": vehicle_type, "Month": month, "DayOfWeek": day_of_week, "Hour": hour, **self._time_flags(day_of_week, hour), "Latitude": latitude, "Longitude": longitude}
        frame = pd.DataFrame([row], columns=self.features)
        pred = float(self.model.predict(self._to_feature_frame(self.preprocessor.transform(frame)))[0])
        if include_time_decay:
            pred *= 0.95
        return max(0.0, float(np.round(pred, 3)))

    def predict_batch(self, *, vehicle_type: str, month: int, day_of_week: int, hour: int, latitude_bins: np.ndarray, longitude_bins: np.ndarray, include_time_decay: bool = False) -> np.ndarray:
        n = len(latitude_bins)
        if len(longitude_bins) != n:
            raise ValueError("latitude_bins and longitude_bins must have the same length")
        if not n:
            return np.array([], dtype=float)
        flags = self._time_flags(day_of_week, hour)
        frame = pd.DataFrame({"Vehicle_Type":[vehicle_type]*n,"Month":[month]*n,"DayOfWeek":[day_of_week]*n,"Hour":[hour]*n,**{k:[v]*n for k,v in flags.items()},"Latitude":latitude_bins.astype(float),"Longitude":longitude_bins.astype(float)}, columns=self.features)
        preds = np.asarray(self.model.predict(self._to_feature_frame(self.preprocessor.transform(frame))), dtype=float)
        preds = np.maximum(0.0, preds)
        if include_time_decay:
            preds *= 0.95
        return np.round(preds, 3)
