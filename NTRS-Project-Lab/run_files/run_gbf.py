from models.utils import load_data
from models.gbf_model import GBFModel
import os
#import cudf  

if __name__ == "__main__":
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df = load_data("data/full_fundamental_dataset.csv")
    #df = cudf.DataFrame(df)  

    model = GBFModel()
    df_forecasts = model.run_forecast(df)

    df_forecasts.to_csv(os.path.join(results_dir, "gbf_forecasts.csv"), index=False)

    print(f"Gradient Boosting model complete. Results saved to {os.path.join(results_dir, 'gbf_forecasts.csv')}")
