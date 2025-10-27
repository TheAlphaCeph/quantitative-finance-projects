# Quantitative Research Portfolio
**University of Chicago | MS Financial Mathematics**  
**Graduating December 2025**

---

## Overview

This repository contains production-quality quantitative trading strategies and research projects developed during my Master's program at the University of Chicago. All projects demonstrate rigorous statistical analysis, sophisticated machine learning implementation, and professional software engineering practices aligned with industry standards.

**Research Focus Areas:**
- Algorithmic Trading & Market Microstructure
- Machine Learning for Financial Forecasting  
- Factor Models & Portfolio Optimization
- Statistical Arbitrage & Pairs Trading
- Risk Analytics & Performance Attribution

---

## Featured Projects

### 1. NLP-Based Momentum Trading Strategy
**Directory:** `momentum_strategy_project/`

#### Summary
Enhanced momentum strategy combining transformer-based NLP sentiment analysis with traditional price momentum signals. Implements "Frog-in-the-Pan" detection algorithm to identify gradual sentiment shifts that persist 3-6 months beyond conventional momentum signals.

#### Technical Implementation
- **Sentiment Analysis:** Transformer models (BERT/RoBERTa) on 120,000+ earnings call transcripts (2010-2024)
- **Signal Construction:** Four-criteria gradual shift detection algorithm
- **Portfolio Optimization:** Mean-variance framework with turnover constraints
- **Backtesting:** Realistic transaction costs, slippage modeling
- **Validation:** Bootstrap confidence intervals, walk-forward analysis

#### Key Results
- **Alpha:** 370 bps annually with statistical significance (t = 2.58, p < 0.05)
- **Sharpe Ratio:** 1.32 (vs. 0.91 for price-only momentum)
- **Factor Independence:** 81% of returns unexplained by Fama-French factors

#### Technologies
`Python` `PyTorch` `Transformers` `CVXPY` `NumPy` `Pandas` `Scikit-learn` `Statsmodels`

#### Documentation
- [Contributions](momentum_strategy_project/CONTRIBUTIONS.md) – My specific research contributions
- [Configuration](momentum_strategy_project/config/) – Strategy parameters and settings

---

### 2. Reinforcement Learning Pair Trading Strategy
**Directory:** `QTS/Final Project/`  

#### Summary
Deep reinforcement learning approach to cryptocurrency pair trading using Proximal Policy Optimization (PPO). Features comprehensive post-strategy analysis with professional risk metrics, capacity constraints, and detailed performance attribution.

#### Critical Analysis & Improvements
- **Identified Issue:** Original RL agent overtrade (1,540 trades in 60 days) → 308% cost drag
- **Root Cause:** Flawed reward function penalized inaction, forcing unprofitable trades
- **Solutions Implemented:**
  - Redesigned reward function with cost awareness
  - Optimal hedge ratio calculation (OLS regression) replacing fixed 1:1 ratios
  - Dynamic position sizing based on volatility regime

#### Technologies
`Python` `OpenAI Gym` `Stable-Baselines3` `PyTorch` `NumPy` `Pandas` `Matplotlib`

#### Documentation
- [Project Analysis](Courses/QTS/Final%20Project/Post%20Strategy%20Analysis.ipynb) – Comprehensive performance review
- [Contributions](Courses/QTS/Final%20Project/CONTRIBUTIONS.md) – My research and implementation work
- [Presentation](Courses/QTS/Final%20Project/Pitch%20Book.pdf) – Executive summary and results

---

### 3. Machine Learning for Earnings Prediction
**Directory:** `NTRS-Project-Lab/`  

#### Summary
Replication and extension of "Fundamental Analysis via Machine Learning" (Cao & You, 2021). Implements ensemble machine learning algorithms to predict earnings changes and generate alpha through factor-based long-short portfolios.

#### Methodology
- **Fama-MacBeth Regressions:** Cross-sectional analysis with industry fixed effects
- **Model Ensemble:** XGBoost, Random Forest, Gradient Boosting, Neural Networks
- **Feature Engineering:** Fundamental characteristics from financial statements
- **Orthogonalization:** Residual earnings component isolated from benchmark forecasts
- **Backtesting:** Transaction cost modeling with realistic implementation friction

#### Technologies
`Python` `XGBoost` `LightGBM` `CatBoost` `TensorFlow` `Keras` `Scikit-learn` `Statsmodels` `Matplotlib` `Seaborn`

#### Documentation
- [Project README](NTRS-Project-Lab/README.md) – Complete methodology
- [Contributions](NTRS-Project-Lab/CONTRIBUTIONS.md) – My quantitative research work
- [Main Script](NTRS-Project-Lab/main.py) – End-to-end implementation

---

## Technical Skills

### Programming & Tools
- **Languages:** Python, C++, R, SQL, Java
- **ML/AI:** PyTorch, TensorFlow, Scikit-learn, XGBoost, LightGBM, Transformers
- **Quant Libraries:** NumPy, Pandas, Statsmodels, Scipy, CVXPY, QuantLib
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Development:** Git, Docker, Jupyter, VS Code, Linux/Unix

### Quantitative Methods
- **Factor Models:** Fama-French, Carhart, Arbitrage Pricing Theory
- **Time Series:** ARIMA, GARCH, Kalman Filters, Cointegration Analysis
- **Machine Learning:** Gradient Boosting, Random Forests, Neural Networks, Reinforcement Learning
- **Optimization:** Convex Optimization, Portfolio Theory, Mean-Variance Analysis
- **Risk Management:** VaR, CVaR, Maximum Drawdown, Tail Risk Metrics

### Statistical Analysis
- **Econometrics:** Panel Regressions, Fixed Effects, Fama-MacBeth Methodology
- **Inference:** Hypothesis Testing, Bootstrap Methods, Cross-Validation
- **Multivariate Analysis:** PCA, Factor Analysis, Dimensionality Reduction

---

## 📁 Repository Structure

```
.
├── momentum_strategy_project/      # NLP-based momentum strategy
│   ├── src/                       # Core implementation modules
│   ├── config/                    # Strategy configuration files
│   ├── tests/                     # Unit and integration tests
│   └── README.md                  # Detailed project documentation
│
├── QTS/Final Project/             # RL pair trading strategy
│   ├── data/                      # Market data and processed features
│   ├── cmds/                      # Command-line utilities
│   ├── images/                    # Performance visualizations
│   ├── results/                   # Backtest results and analysis
│   └── Post_Strategy_Analysis.ipynb  # Comprehensive risk analytics
│
├── NTRS-Project-Lab/              # ML earnings prediction (Northern Trust)
│   ├── data/                      # Financial statement data
│   ├── models/                    # Trained ML models
│   ├── notebooks/                 # Research notebooks
│   ├── gpu_models/                # GPU-accelerated implementations
│   └── main.py                    # Main execution script
│
└── Personal Interest/             # Additional research projects
    ├── Intermarket Prediction Strategy.ipynb
    ├── Quantile Trading.ipynb
    └── Spread Trading Simulation.ipynb
```

---

## 📝 Research Philosophy

My approach to quantitative research emphasizes:

1. **Statistical Rigor:** Every strategy undergoes comprehensive validation including bootstrap confidence intervals, cross-validation, and regime analysis
2. **Production Quality:** All code follows software engineering best practices with proper testing, documentation, and error handling
3. **Realistic Implementation:** Transaction costs, slippage, market impact, and capacity constraints are integral to strategy development
4. **Academic Grounding:** Implementations build on peer-reviewed research while extending methodology with novel approaches
5. **Risk Management:** Comprehensive risk analytics using industry-standard metrics (VaR, CVaR, drawdowns, tail risk)

---

## 🔬 Future Research Directions

- **Multi-Asset Strategies:** Extending momentum framework to commodities and rates markets
- **Alternative Data:** Satellite imagery, web scraping, and unconventional data sources
- **Market Microstructure:** Tick-level analysis and ultra-high-frequency trading signals
- **Deep Learning:** Advanced architectures (Transformers, GRUs) for time series forecasting

---

## ⚖️ Disclaimer

This repository contains academic research projects developed for educational purposes at the University of Chicago. All backtesting results are simulated and may not reflect actual trading performance. Past performance does not guarantee future results. These materials are not investment advice.

---

## 📄 License

This repository is for portfolio demonstration purposes. Code is available for review but not for commercial use without permission.

---

*Last Updated: October 2025*
