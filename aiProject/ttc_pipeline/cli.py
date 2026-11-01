from pathlib import Path
import pandas as pd
from .config import TrainingConfig
from .training import ModelTrainingPipeline


def run(data_path="dataset/ttc_delays_processed.csv", artifacts_dir="server/model_artifacts"):
    data = Path(data_path)
    artifacts = Path(artifacts_dir)
    frame = pd.read_csv(data)
    result = ModelTrainingPipeline(TrainingConfig(data, artifacts)).run(frame)
    print("Best model:", result.best_model_name)
    print("Validation RMSE:", round(result.validation_metrics["rmse"], 3))
    print("Test RMSE:", round(result.test_metrics["rmse"], 3))


if __name__ == "__main__":
    run()
