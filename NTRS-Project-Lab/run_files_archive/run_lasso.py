from models.utils import load_data
from models.lasso_model import LassoModel
import os

if __name__ == "__main__":
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    df = load_data("data/full_fundamental_dataset.csv")
    model = LassoModel()
    df_forecasts = model.run_forecast(df)
    df_forecasts.to_csv(os.path.join(results_dir, "lasso_forecasts.csv"), index=False)
    print(f"LASSO model complete. Results saved to {os.path.join(results_dir, 'lasso_forecasts.csv')}")