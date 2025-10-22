from models.utils import load_data
from models.ann_model import ANNBaggingModel
import os

if __name__ == "__main__":
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    df = load_data("data/prelim_fundamental_dataset.csv")
    model = ANNBaggingModel(use_params=True)
    df_forecasts = model.run_forecast(df)
    df_forecasts.to_csv(os.path.join(results_dir, "ann_forecasts.csv"), index=False)
    print(f"ANN model complete. Results saved to {os.path.join(results_dir, 'ann_forecasts.csv')}")