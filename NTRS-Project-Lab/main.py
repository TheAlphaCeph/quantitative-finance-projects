      
import pandas as pd
import os
from models.utils import load_data, newey_west_tstat
from models.ols_model import OLSModel
from models.random_walk_model import RandomWalkModel
from models.lasso_model import LassoModel
from models.ridge_model import RidgeModel
from models.rf_model import RFModel
from models.gbf_model import GBFModel
from models.ar_model import ARModel
from models.hvz_model import HVZModel
from models.ri_model import RIModel
from models.xgb_model import XGBModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.ann_model import ANNBaggingModel

def run_all_models(df, results_dir="results"):
    """Runs all models and saves results."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    models = [
        OLSModel(),
        RandomWalkModel(),
        ARModel(),
        HVZModel(),
        RIModel(),
        LassoModel(),
        RidgeModel(),
        RFModel(),
        GBFModel(),
        XGBModel(),
        LightGBMModel(),
        CatBoostModel(),
        ANNBaggingModel()
    ]

    for model in models:
        print(f"Running {model.name}...")
        df_forecasts = model.run_forecast(df)
        filepath = os.path.join(results_dir, f"{model.name}_forecasts.csv")
        df_forecasts.to_csv(filepath, index=False)
        print(f"  Saved results to {filepath}")

def load_merge_results(results_dir="full_results"):
    """Loads results from all model CSV files and merges them."""
    df_merged = None
    model_cols = {
        "ols": ["gvkey", "forecast_year", "mkt_cap", "E", "E_future", "abs_error_ols_scaled"],
        "ann": ["gvkey", "forecast_year", "abs_error_ann_bagging_scaled"],
        "ar": ["gvkey", "forecast_year", "abs_error_ar_scaled"],
        "catboost": ["gvkey", "forecast_year", "abs_error_catboost_scaled"],
        "gbf": ["gvkey", "forecast_year", "abs_error_gbf_scaled"],
        "hvz": ["gvkey", "forecast_year", "abs_error_hvz_scaled"],
        "lasso": ["gvkey", "forecast_year", "abs_error_lasso_scaled"],
        "lightgbm": ["gvkey", "forecast_year", "abs_error_lightgbm_scaled"],
        "rf": ["gvkey", "forecast_year", "abs_error_rf_scaled"],
        "ri": ["gvkey", "forecast_year", "abs_error_ri_scaled"],
        "ridge": ["gvkey", "forecast_year", "abs_error_ridge_scaled"],
        "rw": ["gvkey", "forecast_year", "E_pred_rw", "abs_error_rw_scaled"],
        "xgb": ["gvkey", "forecast_year", "abs_error_xgb_scaled"]
    }

    for filename in os.listdir(results_dir):
        if filename.endswith("_forecasts.csv"):
            model_name = filename.replace("_forecasts.csv", "")
            filepath = os.path.join(results_dir, filename)
            usecols = model_cols.get(model_name)
            if usecols is None:
                print(f"Warning: No columns defined for model '{model_name}'. Skipping file '{filename}'.")
                continue
            try:
                df_current = pd.read_csv(filepath, usecols=usecols)
            except ValueError as e:
                print(f"Error reading file '{filename}': {e}. Skipping file.")
                continue

            df_current.drop_duplicates(subset=["gvkey", "forecast_year"], keep="first", inplace=True)

            if df_merged is None:
                df_merged = df_current
            else:
                df_merged = df_merged.merge(df_current, on=["gvkey", "forecast_year"], how="left")

    return df_merged

def compute_summary_statistics(df_forecasts):
    """Computes and prints summary statistics."""
    grouped = df_forecasts.groupby('forecast_year').agg(
      abs_error_ols_mean     = ('abs_error_ols_scaled', 'mean'),
      abs_error_ols_median   = ('abs_error_ols_scaled', 'median'),
      abs_error_ar_mean      = ('abs_error_ar_scaled', 'mean'),
      abs_error_ar_median    = ('abs_error_ar_scaled', 'median'),
      abs_error_hvz_mean     = ('abs_error_hvz_scaled', 'mean'),
      abs_error_hvz_median   = ('abs_error_hvz_scaled', 'median'),
      abs_error_ri_mean      = ('abs_error_ri_scaled', 'mean'),
      abs_error_ri_median    = ('abs_error_ri_scaled', 'median'),
      abs_error_rw_mean      = ('abs_error_rw_scaled', 'mean'),
      abs_error_rw_median    = ('abs_error_rw_scaled', 'median'),
      abs_error_lasso_mean   = ('abs_error_lasso_scaled', 'mean'),
      abs_error_lasso_median = ('abs_error_lasso_scaled', 'median'),
      abs_error_ridge_mean   = ('abs_error_ridge_scaled', 'mean'),
      abs_error_ridge_median = ('abs_error_ridge_scaled', 'median'),
      abs_error_gbf_mean     = ('abs_error_gbf_scaled', 'mean'),
      abs_error_gbf_median   = ('abs_error_gbf_scaled', 'median'),
      abs_error_rf_mean      = ('abs_error_rf_scaled', 'mean'),
      abs_error_rf_median    = ('abs_error_rf_scaled', 'median'),
      abs_error_xgb_mean     = ('abs_error_xgb_scaled', 'mean'),
      abs_error_xgb_median   = ('abs_error_xgb_scaled', 'median'),
      abs_error_lgbm_mean    = ('abs_error_lightgbm_scaled', 'mean'),
      abs_error_lgbm_median  = ('abs_error_lightgbm_scaled', 'median'),
      abs_error_cat_mean     = ('abs_error_catboost_scaled', 'mean'),
      abs_error_cat_median   = ('abs_error_catboost_scaled', 'median'),
      abs_error_ann_mean     = ('abs_error_ann_bagging_scaled', 'mean'),
      abs_error_ann_median   = ('abs_error_ann_bagging_scaled', 'median')
    ).reset_index()


    for model in ['ols', 'ar', 'hvz', 'ri', 'lasso', 'ridge','rf', 'gbf', 'xgb', 'lgbm', 'cat', 'ann']: 
        grouped[f'diff_{model}_mean'] = grouped[f'abs_error_{model}_mean'] - grouped['abs_error_rw_mean']
        grouped[f'diff_{model}_median'] = grouped[f'abs_error_{model}_median'] - grouped['abs_error_rw_median']

    # Compute overall DIFF (time-series averages) for each comparison
    DIFF_metrics = {}
    t_stats = {}
    pctDIFF = {}
    for model in ['ols', 'ar', 'hvz', 'ri', 'lasso', 'ridge', 'gbf', 'rf', 'xgb', 'lgbm', 'cat', 'ann']: 
        DIFF_mean = grouped[f'diff_{model}_mean'].mean()
        DIFF_median = grouped[f'diff_{model}_median'].mean()
        t_mean, _, _ = newey_west_tstat(grouped[f'diff_{model}_mean'], lags=3)
        t_median, _, _ = newey_west_tstat(grouped[f'diff_{model}_median'], lags=3)
        DIFF_metrics[model] = (DIFF_mean, DIFF_median)
        t_stats[model] = (t_mean, t_median)

    # Benchmark averages for percentage differences (using RW error)
    avg_bench_mean   = grouped['abs_error_rw_mean'].mean()
    avg_bench_median = grouped['abs_error_rw_median'].mean()

    for model in ['ols', 'ar', 'hvz', 'ri', 'lasso', 'ridge', 'gbf', 'rf', 'xgb', 'lgbm', 'cat', 'ann']:
        DIFF_mean, DIFF_median = DIFF_metrics[model]
        pctDIFF[model] = (DIFF_mean / avg_bench_mean, DIFF_median / avg_bench_median)
    print("\n=== Forecast Year Summary ===")
    print(grouped.to_string(index=False))

    overall_results = pd.DataFrame({
        "Model": ["OLS", "AR", "HVZ", "RI", "Random Walk", "LASSO", "Ridge", "GBF", "RF", "XGB", "LGBM", "CatBoost", "ANN"], 
        "Mean Absolute Forecast Error": [
            df_forecasts['abs_error_ols_scaled'].mean(),
            df_forecasts['abs_error_ar_scaled'].mean(),
            df_forecasts['abs_error_hvz_scaled'].mean(),
            df_forecasts['abs_error_ri_scaled'].mean(),
            df_forecasts['abs_error_rw_scaled'].mean(),
            df_forecasts['abs_error_lasso_scaled'].mean(),
            df_forecasts['abs_error_ridge_scaled'].mean(),
            df_forecasts['abs_error_gbf_scaled'].mean(),
            df_forecasts['abs_error_rf_scaled'].mean(),
            df_forecasts['abs_error_xgb_scaled'].mean(),
            df_forecasts['abs_error_lightgbm_scaled'].mean(),
            df_forecasts['abs_error_catboost_scaled'].mean(),
            df_forecasts['abs_error_ann_bagging_scaled'].mean()
        ],
        "Median Absolute Forecast Error": [
            df_forecasts['abs_error_ols_scaled'].median(),
            df_forecasts['abs_error_ar_scaled'].median(),
            df_forecasts['abs_error_hvz_scaled'].median(),
            df_forecasts['abs_error_ri_scaled'].median(),
            df_forecasts['abs_error_rw_scaled'].median(),
            df_forecasts['abs_error_lasso_scaled'].median(),
            df_forecasts['abs_error_ridge_scaled'].median(),
            df_forecasts['abs_error_gbf_scaled'].median(),
            df_forecasts['abs_error_rf_scaled'].median(),
            df_forecasts['abs_error_xgb_scaled'].median(),
            df_forecasts['abs_error_lightgbm_scaled'].median(),
            df_forecasts['abs_error_catboost_scaled'].median(),
            df_forecasts['abs_error_ann_bagging_scaled'].median()
        ]
    })

    print("\n=== Overall Forecast Errors (Scaled by mkt_cap) ===")
    print(overall_results.to_string(index=False))

    diff_list = []
    for model in ['ols', 'ar', 'hvz', 'ri', 'lasso', 'ridge', 'rf', 'gbf', 'xgb', 'lgbm', 'cat', 'ann']: 
        DIFF_mean, DIFF_median = DIFF_metrics[model]
        t_mean, t_median = t_stats[model]
        pct_mean, pct_median = pctDIFF[model]
        diff_list.append({
            "Comparison": f"{model.upper()} vs RW",
            "DIFF (mean)": DIFF_mean,
            "Newey–West t-stat (mean)": t_mean,
            "%DIFF (mean)": pct_mean * 100,
            "DIFF (median)": DIFF_median,
            "Newey–West t-stat (median)": t_median,
            "%DIFF (median)": pct_median * 100
        })

    diff_metrics_df = pd.DataFrame(diff_list)

    print("\n=== DIFF Metrics (Model vs. RW) ===")
    print(diff_metrics_df.to_string(index=False))

if __name__ == "__main__":
    #df = load_data("data/prelim_fundamental_dataset.csv")
    #run_all_models(df)  # Run all models and save results
    df_forecasts = load_merge_results()  # Load results
    compute_summary_statistics(df_forecasts)  # Compute and print statistics

    