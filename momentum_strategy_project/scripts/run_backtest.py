#!/usr/bin/env python3
"""
Backtest Execution Script
Momentum Strategy with NLP Sentiment Analysis

Run historical backtests with configurable parameters.
Usage:
    python run_backtest.py --start 2020-01-01 --end 2024-12-31 --tickers AAPL MSFT
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from interfaces.sentiment_api import SentimentAPI
from interfaces.crsp_loader import CRSPLoader
from signals.frog_in_pan_detector import FrogInPanDetector
from signals.signal_constructor import SignalConstructor
from strategy.portfolio_builder import PortfolioBuilder
from strategy.backtester import Backtester
from analysis.performance_metrics import PerformanceAnalyzer
from analysis.factor_attribution import FactorAttribution
from analysis.bootstrap_validation import BootstrapValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path='config/strategy_config.yaml'):
    """Load strategy configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run momentum strategy backtest',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--start', type=str, required=True,
                       help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                       help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, nargs='+',
                       help='Optional ticker list (default: S&P 500)')
    parser.add_argument('--initial-capital', type=float, default=10_000_000,
                       help='Initial portfolio value in USD')
    parser.add_argument('--rebalance-freq', type=str, default='M',
                       choices=['D', 'W', 'M', 'Q'],
                       help='Rebalancing frequency')
    parser.add_argument('--config', type=str, default='config/strategy_config.yaml',
                       help='Path to strategy config file')
    parser.add_argument('--output', type=str, default='output/',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def initialize_data_sources(config):
    """Initialize sentiment API and CRSP data loader"""
    logger.info("Initializing data sources...")
    
    # Load credentials
    with open('config/credentials.yaml', 'r') as f:
        creds = yaml.safe_load(f)
    
    # Initialize sentiment API
    sentiment_api = SentimentAPI(
        api_key=creds['sentiment_api']['key'],
        base_url=creds['sentiment_api']['url']
    )
    
    # Initialize CRSP loader (WRDS connection)
    crsp_loader = CRSPLoader(
        wrds_username=creds['wrds']['username'],
        wrds_password=creds['wrds']['password']
    )
    
    return sentiment_api, crsp_loader


def run_backtest(args, config):
    """Execute full backtest pipeline"""
    
    # Parse dates
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    
    logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial capital: ${args.initial_capital:,.0f}")
    
    # Initialize data sources
    sentiment_api, crsp_loader = initialize_data_sources(config)
    
    # Get universe
    if args.tickers:
        universe = args.tickers
        logger.info(f"Using custom universe: {len(universe)} tickers")
    else:
        universe = crsp_loader.get_sp500_constituents(start_date)
        logger.info(f"Using S&P 500 universe: {len(universe)} tickers")
    
    # Load price data
    logger.info("Loading price data from CRSP...")
    prices = crsp_loader.load_prices(
        tickers=universe,
        start_date=start_date - pd.DateOffset(months=18),  # Extra for momentum calc
        end_date=end_date,
        adjust_for_splits=True,
        adjust_for_dividends=True
    )
    
    # Load sentiment data
    sentiment = sentiment_api.get_sentiment_scores(
        tickers=universe,
        start_date=start_date - pd.DateOffset(months=18),
        end_date=end_date,
        model='roberta-earnings'  # Using RoBERTa trained on earnings calls
    )
    
    # Initialize signal components
    logger.info("Initializing signal generation components...")
    
    frog_detector = FrogInPanDetector(
        lookback_window=config['frog_detector']['lookback_window'],
        trend_threshold=config['frog_detector']['trend_threshold'],
        volatility_threshold=config['frog_detector']['volatility_threshold']
    )
    
    signal_constructor = SignalConstructor(
        price_momentum_weight=config['signal_weights']['price_momentum'],
        sentiment_momentum_weight=config['signal_weights']['sentiment_momentum'],
        frog_pan_weight=config['signal_weights']['frog_pan'],
        factor_orthogonalize=config['signal_constructor']['factor_orthogonalize']
    )
    
    # Generate signals
    logger.info("Generating trading signals...")
    signals = signal_constructor.construct_signals(
        prices=prices,
        sentiment=sentiment,
        frog_detector=frog_detector,
        universe=universe,
        start_date=start_date,
        end_date=end_date
    )
    
    # Build portfolios
    logger.info("Constructing portfolios...")
    portfolio_builder = PortfolioBuilder(
        n_long=config['portfolio']['n_long'],
        n_short=config['portfolio']['n_short'],
        leverage=config['portfolio']['leverage'],
        max_position_size=config['portfolio']['max_position_size'],
        max_sector_exposure=config['portfolio']['max_sector_exposure']
    )
    
    positions = portfolio_builder.build_portfolios(
        signals=signals,
        prices=prices,
        sector_mapping=crsp_loader.get_sector_mapping(universe),
        rebalance_freq=args.rebalance_freq
    )
    
    # Run backtest
    logger.info("Executing backtest...")
    backtester = Backtester(
        initial_capital=args.initial_capital,
        commission_rate=config['costs']['commission_rate'],
        slippage_bps=config['costs']['slippage_bps'],
        borrow_cost_bps=config['costs']['borrow_cost_bps']
    )
    
    results = backtester.run(
        positions=positions,
        prices=prices,
        start_date=start_date,
        end_date=end_date
    )
    
    # Performance analysis
    logger.info("Analyzing performance...")
    perf_analyzer = PerformanceAnalyzer()
    
    metrics = perf_analyzer.calculate_metrics(
        returns=results['returns'],
        positions=positions,
        benchmark_returns=crsp_loader.get_sp500_returns(start_date, end_date)
    )
    
    # Factor attribution
    logger.info("Running factor attribution...")
    factor_attr = FactorAttribution()
    
    factor_loadings, alpha = factor_attr.decompose_returns(
        returns=results['returns'],
        start_date=start_date,
        end_date=end_date
    )
    
    # Bootstrap validation
    logger.info("Running bootstrap validation...")
    bootstrap = BootstrapValidator(n_bootstrap=10000, confidence_level=0.95)
    
    sharpe_ci = bootstrap.validate_sharpe(results['returns'])
    alpha_ci = bootstrap.validate_alpha(results['returns'], start_date, end_date)
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_path}/")
    
    # Portfolio returns
    results['returns'].to_csv(output_path / 'daily_returns.csv')
    results['equity_curve'].to_csv(output_path / 'equity_curve.csv')
    
    # Performance metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path / 'performance_metrics.csv', index=False)
    
    # Factor attribution
    pd.DataFrame([factor_loadings]).to_csv(output_path / 'factor_loadings.csv', index=False)
    pd.DataFrame([{'alpha_annual': alpha}]).to_csv(output_path / 'alpha.csv', index=False)
    
    # Bootstrap confidence intervals
    ci_df = pd.DataFrame({
        'metric': ['sharpe_ratio', 'alpha_annual'],
        'point_estimate': [metrics['sharpe_ratio'], alpha],
        'ci_lower': [sharpe_ci[0], alpha_ci[0]],
        'ci_upper': [sharpe_ci[1], alpha_ci[1]]
    })
    ci_df.to_csv(output_path / 'confidence_intervals.csv', index=False)
    
    # Positions history
    positions.to_csv(output_path / 'positions_history.csv')
    
    # Summary statistics
    summary = {
        'backtest_period': f"{start_date.date()} to {end_date.date()}",
        'trading_days': len(results['returns']),
        'initial_capital': args.initial_capital,
        'final_value': results['equity_curve'].iloc[-1],
        'total_return': (results['equity_curve'].iloc[-1] / args.initial_capital - 1) * 100,
        'cagr': metrics['cagr'] * 100,
        'sharpe_ratio': metrics['sharpe_ratio'],
        'sharpe_95ci': f"[{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]",
        'max_drawdown': metrics['max_drawdown'] * 100,
        'calmar_ratio': metrics['calmar_ratio'],
        'alpha_annual': alpha * 100,
        'alpha_95ci': f"[{alpha_ci[0]*100:.2f}, {alpha_ci[1]*100:.2f}]",
        'avg_turnover': metrics['avg_turnover'] * 100,
        'avg_num_positions': metrics['avg_num_positions']
    }
    
    with open(output_path / 'summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BACKTEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Period:              {summary['backtest_period']}")
    logger.info(f"Total Return:        {summary['total_return']:.2f}%")
    logger.info(f"CAGR:                {summary['cagr']:.2f}%")
    logger.info(f"Sharpe Ratio:        {summary['sharpe_ratio']:.2f} {summary['sharpe_95ci']}")
    logger.info(f"Max Drawdown:        {summary['max_drawdown']:.2f}%")
    logger.info(f"Calmar Ratio:        {summary['calmar_ratio']:.2f}")
    logger.info(f"Annual Alpha:        {summary['alpha_annual']:.2f}% {summary['alpha_95ci']}")
    logger.info(f"Avg Turnover:        {summary['avg_turnover']:.1f}%")
    logger.info(f"Avg # Positions:     {summary['avg_num_positions']:.1f}")
    logger.info("="*60)
    
    return results, metrics


def main():
    """Main execution"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        config = load_config(args.config)
        results, metrics = run_backtest(args, config)
        
        logger.info(f"\n✓ Backtest completed successfully")
        logger.info(f"✓ Results saved to {args.output}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
