from models.utils import load_data
from models.hvz_model import HVZModel
import os

if __name__ == "__main__":
    results_dir = "full_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df = load_data("data/full_fundamental_dataset.csv")
    model = HVZModel()
    df_forecasts = model.run_forecast(df)
    df_forecasts.to_csv(os.path.join(results_dir, "hvz_forecasts.csv"), index=False)
    print(f"HVZ model complete. Results saved to {os.path.join(results_dir, 'hvz_forecasts.csv')}")