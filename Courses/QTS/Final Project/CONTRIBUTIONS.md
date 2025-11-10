# Contributions to Reinforcement Learning Pair Trading Project
## Advanced Statistical Arbitrage Using Deep Reinforcement Learning

---

## Executive Summary

This project developed a sophisticated pair trading strategy using Proximal Policy Optimization (PPO) reinforcement learning applied to cryptocurrency markets. My primary contributions focused on the complete implementation of the RL trading framework, comprehensive post-strategy analysis with industry-standard risk metrics, and rigorous quantitative evaluation that diagnosed critical issues with transaction cost sensitivity.

---

## My Key Contributions

### 1. Comprehensive Post-Strategy Analysis

**Industry-Standard Risk Metrics:**
- Implemented complete suite of risk metrics including Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR), Maximum Drawdown, Sortino Ratio, and Calmar Ratio
- Calculated these metrics across three portfolio configurations and two cost scenarios (zero-cost and realistic costs)

**Transaction Cost Impact Analysis:**
- Conducted detailed analysis of transaction cost effects revealing **300%+ cost drag** from excessive trading frequency
- Identified root cause: RL agent made 1,200-1,500 trades in 60 days (25 trades/day) leading to catastrophic performance degradation
- Demonstrated strategy viability in zero-cost scenario (Sharpe 1.6-2.8) but severe negative returns (-62% to -99%) with realistic costs
- Quantified exact cost attribution: at 0.2% round-trip costs, 1,540 trades consumed 308% of capital

**Trade-Level Analysis:**
- Performed comprehensive trade frequency analysis identifying excessive trading as primary failure mode
- Calculated trade statistics including average holding period (55 minutes), win rate (48%), and profit per trade
- Implemented trade clustering analysis to identify regime-dependent behavior patterns
- Created detailed trade attribution showing return decomposition between strategy alpha and transaction costs

### 2. RL Strategy Implementation

**Pair Selection & Cointegration:**
- Conducted Augmented Dickey-Fuller (ADF) tests on 10 cryptocurrency pairs to identify cointegration relationships
- Implemented Engle-Granger two-step methodology for cointegration testing
- Selected BTC-ETH, BNB-ETH, and ADA-BTC pairs based on statistical significance (p-values < 0.05)
- Validated stationarity of spread series using rolling window tests

**State Space Design:**
- Created comprehensive state representation including spread levels, momentum indicators, and volatility measures
- Implemented proper normalization and standardization of features to improve RL training stability
- Designed action space with discrete position sizing (-1, 0, +1) across multiple pairs
- Built proper state transition logic handling portfolio constraints and risk limits

### 3. Problem Diagnosis & Improvement Roadmap

**Root Cause Analysis:**
- Identified three critical flaws: (1) reward function penalizing inaction forcing unprofitable trades, (2) absence of transaction costs in observation space, (3) fixed 1:1 hedge ratios
- Demonstrated that zero-cost Sharpe of 2.8 proves underlying strategy has edge, but implementation issues destroyed profitability
- Quantified that RL agent failed to learn "no-trade" decision despite it being optimal action ~95% of the time
- Analyzed reward function gradients showing agent received negative feedback for holding positions

**Improvement Framework:**
- Developed detailed priority-ranked list of fixes with expected impact on performance metrics
- **Priority 1 (Critical):** Redesign reward function allowing/encouraging no-trade decisions, expected Sharpe improvement from -27 to 1.2-1.8
- **Priority 2 (Critical):** Implement optimal hedge ratios and add costs to observation space
- **Priority 3 (High):** Reduce target trading frequency by 10-20x through exploration penalty tuning
- Created expected post-fix performance targets: Sharpe 1.2-1.8, annual returns 20-35%, max drawdown <20%

---

## Contact

**Abhay Kanwar**  
M.S. Financial Mathematics, University of Chicago  
abhaykanwar@uchicago.edu | linkedin.com/in/abhaykanwar
