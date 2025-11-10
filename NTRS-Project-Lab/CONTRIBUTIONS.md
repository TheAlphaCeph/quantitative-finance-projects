# Contributions to Northern Trust Asset Management Project
## Quantitative Earnings Prediction Using Machine Learning

**University of Chicago Project Lab | January 2025 - March 2025**  
**Project Partner:** Northern Trust Asset Management  

---

## Executive Summary

This project replicated and extended the academic research "Fundamental Analysis via Machine Learning" (Cao & You, 2021), implementing sophisticated machine learning algorithms to predict earnings changes and generate alpha in equity markets. My primary contributions focused on the quantitative modeling, backtesting infrastructure, and rigorous statistical analysis that formed the core of our deliverable to Northern Trust.

---

## My Key Contributions

### 1. Machine Learning Model Implementation

**Ensemble Framework Development:**
- Implemented comprehensive ensemble modeling framework using XGBoost, Random Forest, ANN, CatBoost, and LightGBM for earnings prediction
- Integrated linear baselines (OLS, LASSO, Ridge) and classical forecasting models (Random Walk, AR) for comparison

**Feature Engineering & Selection:**
- Engineered 56-variable feature set comprising 28 fundamental accounting variables and their first differences for temporal dynamics
- Conducted feature importance analysis identifying critical financial statement variables (operating cash flow, tax expenses, earnings components)
- Created orthogonalized ML residual earnings signals by removing linear model predictions (OLS, LASSO, Ridge) from ensemble forecasts

**Technical Implementation:**
- Wrote Python implementation for large-scale panel data (1975-2024 sample period)
- Implemented parallel hyperparameter optimization (Optuna framework with n_jobs=3) in gradient boosting models
- Built walk-forward validation framework with 10-year rolling training windows to prevent look-ahead bias

### 2. Fama-MacBeth Regression Analysis

**Statistical Methodology:**
- Implemented Fama-MacBeth two-step cross-sectional regression methodology with industry fixed effects
- Conducted rigorous statistical testing including Newey-West standard errors with 6-month lags to account for serial correlation
- Calculated proper t-statistics and performed significance testing across 45-year sample period

**Results & Performance:**
- Generated ML residual signals from ensemble models; best-performing model (CatBoost, value-weighted) achieved **~75 basis points monthly alpha**
- Evaluated performance using Fama-French 5-factor model adjustments across equal-weighted and value-weighted portfolio constructions
- Demonstrated predictive power in earnings forecasting with out-of-sample validation from 2019-2024

**Industry Fixed Effects:**
- Implemented 3-digit SIC code industry controls to isolate stock-specific alpha from sector effects
- Validated robustness of results across different industry classifications
- Ensured proper handling of missing industry data and edge cases

### 3. Comprehensive Backtesting & Validation

**Performance Attribution:**
- Constructed detailed performance attribution framework decomposing returns into alpha sources
- Analyzed strategy performance across different market regimes (2008 crisis, 2020 volatility, normal markets)

**Risk Analysis:**
- Implemented Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) analysis for tail risk assessment
- Performed sensitivity analysis testing strategy robustness to parameter variations

---

## Repository Contents

- **`models/`** - Implementation of ensemble models (XGBoost, RF, ANN, CatBoost, LightGBM) and linear baselines
- **`run_files/`** - Execution scripts for training each model with hyperparameter configurations
- **`notebooks/`** - Fama-MacBeth regression, portfolio analysis, and performance evaluation notebooks
- **`results/`** - Forecast outputs, performance tables, and factor attribution analysis (Panel A/B results)
- **`data/`** - Data loading and preprocessing utilities (WRDS/Compustat queries)
- **`gpu_models/`** - GPU-accelerated model implementations using RAPIDS cuML

---

## Contact & Citations

**Abhay Kanwar**  
M.S. Financial Mathematics, University of Chicago  
abhaykanwar@uchicago.edu | linkedin.com/in/abhaykanwar

**Academic Reference:**  
Cao, J., & You, H. (2021). Fundamental Analysis via Machine Learning. *Working Paper*.
