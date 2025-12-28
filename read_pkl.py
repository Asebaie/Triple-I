import joblib

artifact_path = "artifacts/ae_model_bundle_optuna.pkl"
artifact_data = joblib.load(artifact_path)

print("Keys in artifact_data:", list(artifact_data.keys()))
print("\nModel Architecture Info:", artifact_data["model_architecture_info"])
print("\nBest Val RMSE:", artifact_data["best_val_rmse"])
print("\nBest Params:", artifact_data["best_params"])
