from models.utils import load_data
from models.xgb_model import XGBModel
import os

if __name__ == "__main__":
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    df = load_data("data/full_fundamental_dataset.csv")
    model = XGBModel()
    df_forecasts = model.run_forecast(df)
    df_forecasts.to_csv(os.path.join(results_dir, "xgb_forecasts.csv"), index=False)
    print(f"XGBoost model complete. Results saved to {os.path.join(results_dir, 'xgb_forecasts.csv')}")