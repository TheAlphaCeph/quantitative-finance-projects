#!/usr/bin/env python3
"""
Parameter Optimization Script
Momentum Strategy with NLP Sentiment Analysis

Walk-forward optimization with proper out-of-sample validation.
Usage:
    python optimize_parameters.py --start 2018-01-01 --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import logging
import pandas as pd
import numpy as np
from itertools import product
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from interfaces.sentiment_api import SentimentAPI
from interfaces.crsp_loader import CRSPLoader
from signals.frog_in_pan_detector import FrogInPanDetector
from signals.signal_constructor import SignalConstructor
from strategy.portfolio_builder import PortfolioBuilder
from strategy.backtester import Backtester
from analysis.performance_metrics import PerformanceAnalyzer
from optimization.parameter_optimizer import ParameterOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Optimize momentum strategy parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--start', type=str, required=True,
                       help='Optimization start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                       help='Optimization end date (YYYY-MM-DD)')
    parser.add_argument('--train-months', type=int, default=36,
                       help='Training window in months')
    parser.add_argument('--test-months', type=int, default=6,
                       help='Testing window in months')
    parser.add_argument('--step-months', type=int, default=3,
                       help='Step size for walk-forward in months')
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'calmar_ratio', 'sortino_ratio', 'information_ratio'],
                       help='Optimization target metric')
    parser.add_argument('--config', type=str, default='config/strategy_config.yaml',
                       help='Path to strategy config file')
    parser.add_argument('--output', type=str, default='optimization_results/',
                       help='Output directory for results')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    
    return parser.parse_args()


def define_parameter_grid():
    """Define parameter search space"""
    
    # Signal weight combinations (must sum to 1.0)
    weight_combinations = []
    for w_price in [0.3, 0.35, 0.4, 0.45, 0.5]:
        for w_sentiment in [0.25, 0.3, 0.35, 0.4]:
            w_frog = 1.0 - w_price - w_sentiment
            if 0.15 <= w_frog <= 0.4:
                weight_combinations.append({
                    'price_momentum': w_price,
                    'sentiment_momentum': w_sentiment,
                    'frog_pan': w_frog
                })
    
    param_grid = {
        # Frog-in-Pan detector parameters
        'frog_lookback': [9, 12, 15, 18],
        'frog_trend_threshold': [0.02, 0.03, 0.04, 0.05],
        'frog_volatility_threshold': [0.5, 0.6, 0.7, 0.8],
        
        # Signal construction
        'signal_weights': weight_combinations,
        'factor_orthogonalize': [True, False],
        
        # Portfolio construction
        'n_long': [30, 40, 50],
        'n_short': [20, 30, 40],
        'leverage': [1.0, 1.5, 2.0],
        'max_position_size': [0.03, 0.04, 0.05],
        
        # Signal thresholds
        'long_threshold_percentile': [0.6, 0.65, 0.7],
        'short_threshold_percentile': [0.3, 0.35, 0.4]
    }
    
    return param_grid


def run_single_backtest(params, data_bundle, start_date, end_date, config):
    """Run backtest with specific parameter set"""
    
    try:
        # Unpack data
        prices = data_bundle['prices']
        sentiment = data_bundle['sentiment']
        universe = data_bundle['universe']
        sector_mapping = data_bundle['sector_mapping']
        
        # Initialize components with specific parameters
        frog_detector = FrogInPanDetector(
            lookback_window=params['frog_lookback'],
            trend_threshold=params['frog_trend_threshold'],
            volatility_threshold=params['frog_volatility_threshold']
        )
        
        signal_constructor = SignalConstructor(
            price_momentum_weight=params['signal_weights']['price_momentum'],
            sentiment_momentum_weight=params['signal_weights']['sentiment_momentum'],
            frog_pan_weight=params['signal_weights']['frog_pan'],
            factor_orthogonalize=params['factor_orthogonalize']
        )
        
        # Generate signals
        signals = signal_constructor.construct_signals(
            prices=prices,
            sentiment=sentiment,
            frog_detector=frog_detector,
            universe=universe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Apply thresholds
        long_threshold = signals.quantile(params['long_threshold_percentile'], axis=1)
        short_threshold = signals.quantile(params['short_threshold_percentile'], axis=1)
        
        signals_filtered = signals.copy()
        for date in signals.index:
            mask_long = signals.loc[date] < long_threshold[date]
            mask_short = signals.loc[date] > short_threshold[date]
            signals_filtered.loc[date, mask_long] = np.nan
            signals_filtered.loc[date, mask_short] = np.nan
        
        # Build portfolios
        portfolio_builder = PortfolioBuilder(
            n_long=params['n_long'],
            n_short=params['n_short'],
            leverage=params['leverage'],
            max_position_size=params['max_position_size'],
            max_sector_exposure=config['portfolio']['max_sector_exposure']
        )
        
        positions = portfolio_builder.build_portfolios(
            signals=signals_filtered,
            prices=prices,
            sector_mapping=sector_mapping,
            rebalance_freq='M'
        )
        
        # Run backtest
        backtester = Backtester(
            initial_capital=10_000_000,
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
        
        # Calculate performance
        perf_analyzer = PerformanceAnalyzer()
        metrics = perf_analyzer.calculate_metrics(
            returns=results['returns'],
            positions=positions,
            benchmark_returns=None  # Will use CRSP benchmark
        )
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Backtest failed with params {params}: {str(e)}")
        return None


def walk_forward_optimization(args, config, param_grid):
    """Execute walk-forward optimization"""
    
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    
    logger.info("="*60)
    logger.info("WALK-FORWARD PARAMETER OPTIMIZATION")
    logger.info("="*60)
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Training window: {args.train_months} months")
    logger.info(f"Testing window: {args.test_months} months")
    logger.info(f"Step size: {args.step_months} months")
    logger.info(f"Optimization metric: {args.metric}")
    
    # Initialize data sources
    logger.info("\nInitializing data sources...")
    with open('config/credentials.yaml', 'r') as f:
        creds = yaml.safe_load(f)
    
    sentiment_api = SentimentAPI(
        api_key=creds['sentiment_api']['key'],
        base_url=creds['sentiment_api']['url']
    )
    
    crsp_loader = CRSPLoader(
        wrds_username=creds['wrds']['username'],
        wrds_password=creds['wrds']['password']
    )
    
    # Get universe and data
    universe = crsp_loader.get_sp500_constituents(start_date)
    logger.info(f"Universe: {len(universe)} tickers")
    
    logger.info("\nLoading historical data...")
    prices = crsp_loader.load_prices(
        tickers=universe,
        start_date=start_date - pd.DateOffset(months=24),
        end_date=end_date,
        adjust_for_splits=True,
        adjust_for_dividends=True
    )
    
    sentiment = sentiment_api.get_sentiment_scores(
        tickers=universe,
        start_date=start_date - pd.DateOffset(months=24),
        end_date=end_date,
        model='roberta-earnings'
    )
    
    sector_mapping = crsp_loader.get_sector_mapping(universe)
    
    data_bundle = {
        'prices': prices,
        'sentiment': sentiment,
        'universe': universe,
        'sector_mapping': sector_mapping
    }
    
    # Generate parameter combinations
    logger.info("\nGenerating parameter grid...")
    optimizer = ParameterOptimizer(
        param_grid=param_grid,
        metric=args.metric,
        n_jobs=args.n_jobs
    )
    
    total_combinations = optimizer.get_grid_size()
    logger.info(f"Total parameter combinations: {total_combinations:,}")
    
    # Walk-forward splits
    current_date = start_date
    fold = 0
    results_by_fold = []
    
    while current_date + pd.DateOffset(months=args.train_months + args.test_months) <= end_date:
        fold += 1
        
        train_start = current_date
        train_end = current_date + pd.DateOffset(months=args.train_months)
        test_start = train_end
        test_end = train_end + pd.DateOffset(months=args.test_months)
        
        logger.info("\n" + "-"*60)
        logger.info(f"FOLD {fold}")
        logger.info(f"Training:  {train_start.date()} to {train_end.date()}")
        logger.info(f"Testing:   {test_start.date()} to {test_end.date()}")
        logger.info("-"*60)
        
        # Optimize on training set
        logger.info("Optimizing parameters on training set...")
        best_params_train = optimizer.optimize(
            data_bundle=data_bundle,
            start_date=train_start,
            end_date=train_end,
            config=config,
            run_backtest_func=run_single_backtest
        )
        
        train_metrics = run_single_backtest(
            params=best_params_train,
            data_bundle=data_bundle,
            start_date=train_start,
            end_date=train_end,
            config=config
        )
        
        logger.info(f"Best training {args.metric}: {train_metrics[args.metric]:.3f}")
        
        # Test on out-of-sample period
        logger.info("Evaluating on out-of-sample test set...")
        test_metrics = run_single_backtest(
            params=best_params_train,
            data_bundle=data_bundle,
            start_date=test_start,
            end_date=test_end,
            config=config
        )
        
        logger.info(f"Test {args.metric}: {test_metrics[args.metric]:.3f}")
        logger.info(f"Test Sharpe: {test_metrics['sharpe_ratio']:.3f}")
        logger.info(f"Test CAGR: {test_metrics['cagr']*100:.2f}%")
        logger.info(f"Test MaxDD: {test_metrics['max_drawdown']*100:.2f}%")
        
        # Store results
        fold_result = {
            'fold': fold,
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'best_params': best_params_train,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        results_by_fold.append(fold_result)
        
        # Move to next window
        current_date += pd.DateOffset(months=args.step_months)
    
    return results_by_fold


def save_results(results, output_path, args):
    """Save optimization results"""
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    with open(output_path / 'optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary DataFrame
    summary_data = []
    for fold_result in results:
        test_metrics = fold_result['test_metrics']
        summary_data.append({
            'fold': fold_result['fold'],
            'test_start': fold_result['test_start'],
            'test_end': fold_result['test_end'],
            'sharpe_ratio': test_metrics['sharpe_ratio'],
            'cagr': test_metrics['cagr'],
            'max_drawdown': test_metrics['max_drawdown'],
            'calmar_ratio': test_metrics['calmar_ratio'],
            'sortino_ratio': test_metrics['sortino_ratio'],
            'avg_turnover': test_metrics['avg_turnover']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / 'fold_summary.csv', index=False)
    
    # Aggregate statistics
    aggregate_stats = {
        'n_folds': len(results),
        'avg_sharpe': summary_df['sharpe_ratio'].mean(),
        'std_sharpe': summary_df['sharpe_ratio'].std(),
        'avg_cagr': summary_df['cagr'].mean(),
        'avg_max_dd': summary_df['max_drawdown'].mean(),
        'avg_calmar': summary_df['calmar_ratio'].mean(),
        'win_rate': (summary_df['sharpe_ratio'] > 1.0).mean()
    }
    
    with open(output_path / 'aggregate_stats.yaml', 'w') as f:
        yaml.dump(aggregate_stats, f, default_flow_style=False)
    
    # Parameter stability analysis
    param_counts = {}
    for fold_result in results:
        params_str = json.dumps(fold_result['best_params'], sort_keys=True)
        param_counts[params_str] = param_counts.get(params_str, 0) + 1
    
    stability_df = pd.DataFrame([
        {'params': k, 'frequency': v} 
        for k, v in sorted(param_counts.items(), key=lambda x: -x[1])
    ])
    stability_df.to_csv(output_path / 'parameter_stability.csv', index=False)
    
    return summary_df, aggregate_stats


def main():
    """Main execution"""
    args = parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Define parameter grid
        param_grid = define_parameter_grid()
        
        # Run walk-forward optimization
        results = walk_forward_optimization(args, config, param_grid)
        
        # Save results
        summary_df, aggregate_stats = save_results(results, args.output, args)
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Number of folds:     {aggregate_stats['n_folds']}")
        logger.info(f"Avg Sharpe Ratio:    {aggregate_stats['avg_sharpe']:.3f} ± {aggregate_stats['std_sharpe']:.3f}")
        logger.info(f"Avg CAGR:            {aggregate_stats['avg_cagr']*100:.2f}%")
        logger.info(f"Avg Max Drawdown:    {aggregate_stats['avg_max_dd']*100:.2f}%")
        logger.info(f"Avg Calmar Ratio:    {aggregate_stats['avg_calmar']:.3f}")
        logger.info(f"Win Rate (Sharpe>1): {aggregate_stats['win_rate']*100:.1f}%")
        logger.info("="*60)
        logger.info(f"\n✓ Results saved to {args.output}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
