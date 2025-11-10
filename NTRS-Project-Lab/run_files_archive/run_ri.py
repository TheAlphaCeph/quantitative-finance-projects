import os
from models.utils import load_data
from models.ri_model import RIModel

if __name__ == "__main__":
    results_dir = "full_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load your fundamental dataset
    df = load_data("data/full_fundamental_dataset.csv")

    # Initialize and run the RI model
    model = RIModel()
    df_forecasts = model.run_forecast(df)

    # Save the forecast results
    output_path = os.path.join(results_dir, "ri_forecasts.csv")
    df_forecasts.to_csv(output_path, index=False)

    print(f"RI model complete. Results saved to {output_path}")
