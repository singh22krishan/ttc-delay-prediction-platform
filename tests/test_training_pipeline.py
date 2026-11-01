import pandas as pd

from aiProject.ttc_pipeline.training import ModelTrainingPipeline, regression_metrics


def sample_frame():
    return pd.DataFrame({
        "Date": ["2023-01-01", "2023-06-01", "2024-02-01", "2024-08-01", "2025-03-01", "2025-09-01"],
        "Time": ["08:00", "17:00", "09:00", "23:00", "07:00", "16:00"],
        "Min Delay": [2, 7, 5, 12, 4, 9],
        "Vehicle_Type": ["BUS", "STREETCAR", "BUS", "SUBWAY", "BUS", "STREETCAR"],
        "Latitude": [43.65, 43.66, 43.67, 43.68, 43.69, 43.70],
        "Longitude": [-79.38, -79.39, -79.40, -79.41, -79.42, -79.43],
    })


def test_prepare_dataframe_adds_inference_features():
    out = ModelTrainingPipeline.prepare_dataframe(sample_frame())
    assert {"Year", "Month", "DayOfWeek", "Hour", "IsWeekend", "IsRushHour", "IsPeakHour", "IsNight"}.issubset(out.columns)
    assert out["Min Delay"].min() >= 0


def test_temporal_split_preserves_year_order():
    prepared = ModelTrainingPipeline.prepare_dataframe(sample_frame())
    train, val, test = ModelTrainingPipeline.temporal_split(prepared)
    assert set(train["Year"]) == {2023}
    assert set(val["Year"]) == {2024}
    assert set(test["Year"]) == {2025}


def test_regression_metrics_zero_for_perfect_predictions():
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["r2"] == 1
