# ML Training Pipeline

The original project contains both notebook-based exploration and an object-oriented training pipeline. The production workflow:

1. Loads and validates the unified TTC delay dataset.
2. Cleans missing values and engineers calendar/geographic features.
3. Creates temporal train, validation, and test partitions.
4. Caps extreme targets using training-only statistics.
5. Builds inference-safe preprocessing using only features available at prediction time.
6. Trains LightGBM and XGBoost regressors.
7. Compares validation/test performance and selects the best model.
8. Retrains the selected model on train + validation data.
9. Exports a `model.pkl` bundle containing the model, preprocessor, feature list, and selected-model name.
10. Exports `heatmap_inference_config.json` containing geographic bins and context indexes used by the API.

Generated model/data artifacts are intentionally excluded from this portfolio repository.
