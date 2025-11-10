import polars as pl
import pandas as pd
import statsmodels.api as sm
import numpy as np
from typing import List, Dict, Tuple

# --- Helper Functions ---

def load_ff3_factors_local(filepath="F-F_Research_Data_Factors.CSV"):
    """Loads and processes the Fama-French 3-factor file (plus RF)."""
    df = pd.read_csv(filepath, skiprows=3)
    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    df = df[df["Date"].astype(str).str.match(r'^\d{6}$')]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100
    df.rename(columns={"Mkt-RF": "Mkt_RF"}, inplace=True)
    return df

def load_ff5_factors_local(filepath="F-F_Research_Data_5_Factors_2x3.CSV"):
    """Loads and processes the Fama-French 5-factor file (plus RF)."""
    df = pd.read_csv(filepath, skiprows=3)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df = df[df["Date"].astype(str).str.match(r'^\d{6}$')]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df.rename(columns={"Mkt-RF": "Mkt_RF"}, inplace=True)
    factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    for col in factor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100
    return df

def load_momentum_factor_local(filepath="F-F_Momentum_Factor.CSV"):
    """Loads and processes the Momentum factor file."""
    df = pd.read_csv(filepath, skiprows=13)
    df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df = df[df["Date"].astype(str).str.match(r'^\d{6}$')]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df["Mom"] = pd.to_numeric(df["Mom"], errors="coerce") / 100
    return df

def create_quintile_portfolios_direct(df: pl.DataFrame, sort_col: str) -> pl.DataFrame:
    """Assigns stocks to quintiles based on the specified sorting column within 3-digit SIC groups."""
    df = df.with_columns(
        pl.col("sic").cast(pl.Utf8).str.slice(0, 3).alias("sic_3d")
    )

    bounds = (
        df
        .group_by(["date", "sic_3d"])
        .agg([
            pl.col(sort_col).quantile(0.2).alias("q20"),
            pl.col(sort_col).quantile(0.4).alias("q40"),
            pl.col(sort_col).quantile(0.6).alias("q60"),
            pl.col(sort_col).quantile(0.8).alias("q80")
        ])
    )

    df = df.join(bounds, on=["date", "sic_3d"], how="left")

    df = df.with_columns(
        pl.when(pl.col(sort_col) <= pl.col("q20")).then(1)
        .when(pl.col(sort_col) <= pl.col("q40")).then(2)
        .when(pl.col(sort_col) <= pl.col("q60")).then(3)
        .when(pl.col(sort_col) <= pl.col("q80")).then(4)
        .otherwise(5)
        .alias("quintile")
    )
    return df

def calculate_portfolio_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates equal- and value-weighted portfolio returns and hedge returns."""
    df_portfolio_agg = df.group_by(["date", "quintile"]).agg([
        pl.col("retadj").mean().alias("ret_ew"),
        ((pl.col("retadj") * pl.col("mkt_cap")).sum() / pl.col("mkt_cap").sum()).alias("ret_vw")
    ])

    df_q1 = df_portfolio_agg.filter(pl.col("quintile") == 1) \
        .select(["date", "ret_ew", "ret_vw"]) \
        .rename({"ret_ew": "ret_ew_1", "ret_vw": "ret_vw_1"})
    df_q5 = df_portfolio_agg.filter(pl.col("quintile") == 5) \
        .select(["date", "ret_ew", "ret_vw"]) \
        .rename({"ret_ew": "ret_ew_5", "ret_vw": "ret_vw_5"})

    df_portfolio = df_q1.join(df_q5, on="date", how="full")

    # Can try changing this part the other way around to see the effect of the value premium
    # quintile 1 - quintile 5 is the value portfolio
    # quintile 5 - quintile 1 is the growth portfolio
    df_portfolio = df_portfolio.with_columns([
        (pl.col("ret_ew_1") - pl.col("ret_ew_5")).alias("hedge_ret_ew"),
        (pl.col("ret_vw_1") - pl.col("ret_vw_5")).alias("hedge_ret_vw")
    ])
    return df_portfolio

def calc_alpha(df: pl.DataFrame, factor_list: List[str], ret_col: str) -> Tuple[float, float, sm.OLS]:
    """Calculates alpha and its p-value from a factor model regression."""
    X = df.select(factor_list).to_pandas()
    X = sm.add_constant(X)
    y = df.select(ret_col).to_pandas().squeeze()
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    alpha = model.params["const"]
    alpha_pvalue = model.pvalues["const"]
    return alpha, alpha_pvalue, model

def calculate_performance_metrics(returns: pl.Series) -> Dict[str, float]:
    """Calculates annualized return, Sharpe ratio, and max drawdown."""
    returns_np = returns.to_numpy()
    annualized_return = (1 + returns_np.mean()) ** 12 - 1
    annualized_std = returns_np.std() * np.sqrt(12)
    if annualized_std > 0:
        annualized_sharpe = annualized_return / annualized_std
    else:
        annualized_sharpe = np.nan
    cumulative_returns = (1 + returns_np).cumprod()
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "annualized_return": annualized_return,
        "annualized_sharpe": annualized_sharpe,
        "max_drawdown": max_drawdown,
    }

# --- Main Pipeline ---

def main():
    # Load data
    df_forecasts = pl.read_parquet("full_results/composite_forecasts.parquet")
    df_fundamental = pl.read_parquet("data/monthly_dataset.parquet")

    # Prepare data
    df_forecasts = df_forecasts.with_columns(
        pl.col("gvkey").cast(pl.Int64).cast(pl.Utf8).str.zfill(6)
    )
    df_fundamental = df_fundamental.with_columns(
        pl.col("gvkey").cast(pl.Int64).cast(pl.Utf8).str.zfill(6),
        pl.col("date").cast(pl.Date),
        pl.col("retadj_x").alias("retadj")
    )
    df_fundamental_subset = df_fundamental.select(["gvkey", "date", "sic", "retadj"])
    df_fundamental_subset = df_fundamental_subset.filter(pl.col("date").dt.year() != 2024)

    df_calendar = (
        df_fundamental_subset.select(
            pl.col("date"),
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month")
        )
        .unique()
        .group_by(["year", "month"])
        .agg(pl.col("date").max().alias("date"))
        .sort(["year", "month"])
        .select(["year", "date"])
    )

    df_forecasts = df_forecasts.with_columns(
        pl.col("year").cast(pl.Int32)
    )
    df_calendar = df_calendar.with_columns(
        pl.col("year").cast(pl.Int32)
    )

    df_forecasts_monthly = df_forecasts.join(
        df_calendar, on="year", how="inner"
    )

    df_merged = df_forecasts_monthly.join(
        df_fundamental_subset, on=["date", "gvkey"], how="inner"
    )
    df_merged = df_merged.drop_nulls()

    # Define forecast columns (ML models and baselines)
    forecast_cols = [
        'E_pred_ols', 'E_pred_ann_bagging', 'E_pred_ar', 'E_pred_catboost', 'E_pred_gbf',
        'E_pred_rf', 'E_pred_hvz', 'E_pred_lasso', 'E_pred_lightgbm', 'E_pred_ri', 'E_pred_ridge',
        'E_pred_rw', 'E_pred_xgb', 'COMP_EXT', 'COMP_LR', 'COMP_NL', 'COMP_ML', 'COMP_MED_EXT',
        'COMP_MED_LR', 'COMP_MED_NL', 'COMP_MED_ML'
    ]

    # Load factor data
    df_ff3 = load_ff3_factors_local("data/F-F_Research_Data_Factors.CSV")
    ff5_df = load_ff5_factors_local("data/F-F_Research_Data_5_Factors_2x3.CSV")
    mom_df = load_momentum_factor_local("data/F-F_Momentum_Factor.CSV")
    df_ff3_pl = pl.from_pandas(df_ff3)
    mom_pl = pl.from_pandas(mom_df)
    df_carhart = df_ff3_pl.join(mom_pl, on="Date", how="inner")
    df_ff5_pl = pl.from_pandas(ff5_df)

    # --- Store results ---
    all_results = []
    count = 1
    for sort_col in forecast_cols:
        print(f"Processing {sort_col} ({count}/{len(forecast_cols)})")
        count += 1

        # Direct quintile sort on the forecast signal
        df_with_quintiles = create_quintile_portfolios_direct(df_merged, sort_col)

        # Calculate portfolio returns (market cap transformation via value-weighting)
        df_portfolio = calculate_portfolio_returns(df_with_quintiles)

        # Adjust date for merging with factor data
        df_portfolio = df_portfolio.with_columns(pl.col("date").cast(pl.Datetime("ns")))
        df_portfolio = df_portfolio.with_columns(pl.col("date").dt.truncate("1mo"))

        # Merge with factor data and compute excess returns
        df_merged_ff3 = df_portfolio.join(df_ff3_pl, left_on="date", right_on="Date", how="inner")
        df_merged_ff3 = df_merged_ff3.with_columns(
            (pl.col("hedge_ret_vw") - pl.col("RF")).alias("excess_ret_vw"),
            (pl.col("hedge_ret_ew") - pl.col("RF")).alias("excess_ret_ew")
        )
        df_reg_carhart = df_portfolio.join(df_carhart, left_on="date", right_on="Date", how="inner")
        df_reg_carhart = df_reg_carhart.with_columns(
            (pl.col("hedge_ret_vw") - pl.col("RF")).alias("excess_ret_vw"),
            (pl.col("hedge_ret_ew") - pl.col("RF")).alias("excess_ret_ew")
        )
        df_merged_ff5 = df_portfolio.join(df_ff5_pl, left_on="date", right_on="Date", how="inner")
        df_merged_ff5 = df_merged_ff5.with_columns(
            (pl.col("hedge_ret_vw") - pl.col("RF")).alias("excess_ret_vw"),
            (pl.col("hedge_ret_ew") - pl.col("RF")).alias("excess_ret_ew")
        )

        # Calculate alphas for factor models
        alpha_vw_capm, pval_vw_capm, _ = calc_alpha(df_merged_ff3, ["Mkt_RF"], "excess_ret_vw")
        alpha_ew_capm, pval_ew_capm, _ = calc_alpha(df_merged_ff3, ["Mkt_RF"], "excess_ret_ew")
        alpha_vw_ff3, pval_vw_ff3, _ = calc_alpha(df_merged_ff3, ["Mkt_RF", "SMB", "HML"], "excess_ret_vw")
        alpha_ew_ff3, pval_ew_ff3, _ = calc_alpha(df_merged_ff3, ["Mkt_RF", "SMB", "HML"], "excess_ret_ew")
        alpha_vw_carhart, pval_vw_carhart, _ = calc_alpha(df_reg_carhart, ["Mkt_RF", "SMB", "HML", "Mom"], "excess_ret_vw")
        alpha_ew_carhart, pval_ew_carhart, _ = calc_alpha(df_reg_carhart, ["Mkt_RF", "SMB", "HML", "Mom"], "excess_ret_ew")
        alpha_vw_ff5, pval_vw_ff5, _ = calc_alpha(df_merged_ff5, ["Mkt_RF", "SMB", "HML", "RMW", "CMA"], "excess_ret_vw")
        alpha_ew_ff5, pval_ew_ff5, _ = calc_alpha(df_merged_ff5, ["Mkt_RF", "SMB", "HML", "RMW", "CMA"], "excess_ret_ew")

        # Calculate performance metrics
        perf_vw = calculate_performance_metrics(df_portfolio["hedge_ret_vw"])
        perf_ew = calculate_performance_metrics(df_portfolio["hedge_ret_ew"])

        # Store results
        results = {
            "sort_col": sort_col,
            "alpha_vw_capm": alpha_vw_capm,
            "pval_vw_capm": pval_vw_capm,
            "alpha_ew_capm": alpha_ew_capm,
            "pval_ew_capm": pval_ew_capm,
            "alpha_vw_ff3": alpha_vw_ff3,
            "pval_vw_ff3": pval_vw_ff3,
            "alpha_ew_ff3": alpha_ew_ff3,
            "pval_ew_ff3": pval_ew_ff3,
            "alpha_vw_carhart": alpha_vw_carhart,
            "pval_vw_carhart": pval_vw_carhart,
            "alpha_ew_carhart": alpha_ew_carhart,
            "pval_ew_carhart": pval_ew_carhart,
            "alpha_vw_ff5": alpha_vw_ff5,
            "pval_vw_ff5": pval_vw_ff5,
            "alpha_ew_ff5": alpha_ew_ff5,
            "pval_ew_ff5": pval_ew_ff5,
            "perf_vw_return": perf_vw["annualized_return"],
            "perf_vw_sharpe": perf_vw["annualized_sharpe"],
            "perf_vw_drawdown": perf_vw["max_drawdown"],
            "perf_ew_return": perf_ew["annualized_return"],
            "perf_ew_sharpe": perf_ew["annualized_sharpe"],
            "perf_ew_drawdown": perf_ew["max_drawdown"],
        }
        all_results.append(results)

    # Save results
    results_df = pl.DataFrame(all_results)
    results_df.write_parquet("full_results/portfolio_analysis_direct_sort.parquet")
    print("Results saved to full_results/portfolio_analysis_direct_sort.parquet")
    return results_df

if __name__ == "__main__":
    results_df = main()