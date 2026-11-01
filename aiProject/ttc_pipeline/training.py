from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from .config import TrainingConfig


FEATURES = [
    "Vehicle_Type", "Month", "Hour", "DayOfWeek", "IsWeekend",
    "IsRushHour", "IsPeakHour", "IsNight", "Latitude", "Longitude",
]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "abs_error_p90": float(np.percentile(np.abs(y_true - y_pred), 90)),
    }


@dataclass
class TrainingResult:
    best_model_name: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


class ModelTrainingPipeline:
    """Portfolio-safe extraction of the original capstone training workflow."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    @staticmethod
    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        required = {"Date", "Time", "Min Delay", "Vehicle_Type", "Latitude", "Longitude"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        out = df.copy()
        out["EventDateTime"] = pd.to_datetime(
            out["Date"].astype(str) + " " + out["Time"].astype(str), errors="coerce"
        )
        out["Min Delay"] = pd.to_numeric(out["Min Delay"], errors="coerce")
        out = out.dropna(subset=["EventDateTime", "Min Delay", "Latitude", "Longitude"]).copy()
        out["Min Delay"] = out["Min Delay"].clip(lower=0)
        out["Year"] = out["EventDateTime"].dt.year
        out["Month"] = out["EventDateTime"].dt.month
        out["DayOfWeek"] = out["EventDateTime"].dt.dayofweek
        out["Hour"] = out["EventDateTime"].dt.hour
        out["IsWeekend"] = out["DayOfWeek"].isin([5, 6]).astype(int)
        out["IsRushHour"] = out["Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
        out["IsPeakHour"] = out["Hour"].isin([6, 7, 8, 9, 15, 16, 17, 18, 19]).astype(int)
        out["IsNight"] = out["Hour"].isin([0, 1, 2, 3, 4, 5, 23]).astype(int)
        return out

    @staticmethod
    def temporal_split(df: pd.DataFrame):
        train = df[df["Year"] <= 2023].copy()
        val = df[df["Year"] == 2024].copy()
        test = df[df["Year"] == 2025].copy()
        if min(len(train), len(val), len(test)) == 0:
            ordered = df.sort_values("EventDateTime").reset_index(drop=True)
            n = len(ordered)
            train_end, val_end = int(n * 0.70), int(n * 0.85)
            train, val, test = ordered.iloc[:train_end], ordered.iloc[train_end:val_end], ordered.iloc[val_end:]
        if min(len(train), len(val), len(test)) == 0:
            raise ValueError("Dataset is too small for train/validation/test splitting")
        cap = float(train["Min Delay"].quantile(0.995))
        return tuple(part.assign(**{"Min Delay": part["Min Delay"].clip(upper=cap)}) for part in (train, val, test))

    @staticmethod
    def build_preprocessor() -> ColumnTransformer:
        return ColumnTransformer([
            ("vehicle", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), ["Vehicle_Type"]),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), FEATURES[1:]),
        ])

    def run(self, df: pd.DataFrame | None = None) -> TrainingResult:
        if df is None:
            if not self.config.processed_file.exists():
                raise FileNotFoundError(self.config.processed_file)
            df = pd.read_csv(self.config.processed_file)

        frame = self.prepare_dataframe(df)
        train, val, test = self.temporal_split(frame)
        pre = self.build_preprocessor()
        x_train = pre.fit_transform(train[FEATURES])
        x_val = pre.transform(val[FEATURES])
        x_test = pre.transform(test[FEATURES])

        candidates = {
            "lightgbm": LGBMRegressor(
                n_estimators=700, learning_rate=0.03, num_leaves=63,
                subsample=0.85, colsample_bytree=0.85,
                random_state=self.config.random_seed, objective="regression", n_jobs=-1,
            ),
            "xgboost": XGBRegressor(
                n_estimators=700, learning_rate=0.03, max_depth=8,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                random_state=self.config.random_seed, objective="reg:squarederror", n_jobs=-1,
            ),
        }

        scores = {}
        for name, model in candidates.items():
            model.fit(x_train, train["Min Delay"])
            scores[name] = {
                "validation": regression_metrics(val["Min Delay"], model.predict(x_val)),
                "test": regression_metrics(test["Min Delay"], model.predict(x_test)),
            }

        best_name = min(scores, key=lambda name: scores[name]["validation"]["rmse"])
        best_model = candidates[best_name]

        trainval = pd.concat([train, val], ignore_index=True)
        final_pre = self.build_preprocessor()
        x_trainval = final_pre.fit_transform(trainval[FEATURES])
        final_model = best_model.__class__(**best_model.get_params())
        final_model.fit(x_trainval, trainval["Min Delay"])

        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": final_model,
            "preprocessor": final_pre,
            "features": FEATURES,
            "best_model_name": best_name,
            "validation_metrics": scores[best_name]["validation"],
            "test_metrics_before_refit": scores[best_name]["test"],
        }
        joblib.dump(bundle, self.config.artifacts_dir / "model.pkl")
        (self.config.artifacts_dir / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

        return TrainingResult(best_name, scores[best_name]["validation"], scores[best_name]["test"])
