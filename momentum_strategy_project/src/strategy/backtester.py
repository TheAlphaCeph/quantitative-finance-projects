"""
Strategy Backtester

Historical simulation with transaction costs and realistic execution.

DECISION CRITERIA:
    1. Construct portfolios at each rebalance date
    2. Calculate returns between rebalances
    3. Apply transaction costs on trades
    4. Track performance metrics over time

INPUTS:
    - signals: Multi-index DataFrame with momentum signals
    - prices: Multi-index DataFrame with stock prices
    - transaction_cost_bps: Costs per trade in basis points
    - rebalance_frequency: 'daily', 'weekly', 'monthly'

OUTPUTS:
    - Strategy returns (time series)
    - Trade history
    - Performance metrics

ASSUMPTIONS:
    - Market orders with immediate execution
    - Transaction costs applied symmetrically (buy and sell)
    - No market impact beyond specified costs
    - Rebalancing at market close prices
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import timedelta
from loguru import logger

from .portfolio_builder import PortfolioBuilder


class Backtester:
    """
    Backtest momentum strategy with realistic execution.
    
    Parameters
    ----------
    transaction_cost_bps : float, optional
        Transaction costs in basis points (default: 10)
    rebalance_frequency : str, optional
        Rebalancing frequency: 'daily', 'weekly', 'monthly' (default: 'monthly')
    initial_capital : float, optional
        Starting capital (default: 10,000,000)
    portfolio_builder : PortfolioBuilder, optional
        Portfolio construction instance (default: creates new)
    
    Examples
    --------
    >>> backtester = Backtester(
    ...     transaction_cost_bps=10,
    ...     rebalance_frequency='monthly',
    ...     initial_capital=10_000_000
    ... )
    >>> 
    >>> results = backtester.run(
    ...     signals=signal_data,
    ...     prices=price_data,
    ...     start_date='2015-01-01',
    ...     end_date='2024-12-31'
    ... )
    """
    
    def __init__(
        self,
        transaction_cost_bps: float = 10,
        rebalance_frequency: str = 'monthly',
        initial_capital: float = 10_000_000,
        portfolio_builder: Optional[PortfolioBuilder] = None
    ):
        self.transaction_cost_bps = transaction_cost_bps
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital
        
        # Initialize portfolio builder if not provided
        self.portfolio_builder = portfolio_builder or PortfolioBuilder()
        
        logger.info(
            f"Backtester initialized: "
            f"cost={transaction_cost_bps}bps, freq={rebalance_frequency}"
        )
    
    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Run backtest simulation.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Multi-index (date, ticker) with composite_signal column
        prices : pd.DataFrame
            Multi-index (date, ticker) with close prices
        start_date : str
            Backtest start date 'YYYY-MM-DD'
        end_date : str
            Backtest end date 'YYYY-MM-DD'
        
        Returns
        -------
        dict
            Backtest results including:
            - returns: Daily strategy returns
            - equity_curve: Portfolio value over time
            - positions: Position history
            - trades: Trade log
            - metrics: Performance summary
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Convert dates
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(
            signals,
            start_date,
            end_date
        )
        
        logger.info(f"Identified {len(rebalance_dates)} rebalance dates")
        
        # Initialize tracking
        portfolio_value = self.initial_capital
        current_weights = {}
        
        equity_curve = []
        position_history = []
        trade_log = []
        daily_returns = []
        
        # Simulation loop
        for i, rebalance_date in enumerate(rebalance_dates):
            logger.debug(f"Processing rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}")
            
            # Build new portfolio
            new_weights = self.portfolio_builder.build_portfolio(
                signals=signals,
                date=rebalance_date
            )
            
            if not new_weights:
                logger.warning(f"No portfolio constructed for {rebalance_date}")
                continue
            
            # Calculate turnover and transaction costs
            turnover = self.portfolio_builder.calculate_turnover(
                current_weights,
                new_weights
            )
            
            transaction_cost = self._calculate_transaction_cost(
                turnover,
                portfolio_value
            )
            
            # Apply transaction costs
            portfolio_value -= transaction_cost
            
            # Record trade
            trade_log.append({
                'date': rebalance_date,
                'turnover': turnover,
                'transaction_cost': transaction_cost,
                'portfolio_value': portfolio_value
            })
            
            # Record positions
            position_history.append({
                'date': rebalance_date,
                'positions': new_weights.copy(),
                'n_positions': len(new_weights)
            })
            
            # Calculate returns until next rebalance
            if i < len(rebalance_dates) - 1:
                next_rebalance = rebalance_dates[i + 1]
                
                period_returns = self._calculate_period_returns(
                    weights=new_weights,
                    prices=prices,
                    start_date=rebalance_date,
                    end_date=next_rebalance
                )
                
                # Update portfolio value and track returns
                for ret_date, ret_value in period_returns.items():
                    portfolio_value *= (1 + ret_value)
                    daily_returns.append({
                        'date': ret_date,
                        'return': ret_value,
                        'portfolio_value': portfolio_value
                    })
            
            # Update current weights
            current_weights = new_weights
        
        # Convert to DataFrames
        returns_df = pd.DataFrame(daily_returns).set_index('date')
        trades_df = pd.DataFrame(trade_log)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(returns_df['return'])
        
        logger.info(
            f"Backtest complete: "
            f"final_value=${portfolio_value:,.0f}, "
            f"total_return={metrics['total_return']:.2%}"
        )
        
        return {
            'returns': returns_df['return'],
            'equity_curve': returns_df['portfolio_value'],
            'positions': position_history,
            'trades': trades_df,
            'metrics': metrics,
            'initial_capital': self.initial_capital,
            'final_capital': portfolio_value
        }
    
    def _get_rebalance_dates(
        self,
        signals: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """
        Get rebalance dates based on frequency.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Signals with date index
        start_date : pd.Timestamp
            Start date
        end_date : pd.Timestamp
            End date
        
        Returns
        -------
        pd.DatetimeIndex
            Rebalance dates
        """
        available_dates = signals.index.get_level_values('date').unique()
        available_dates = available_dates[
            (available_dates >= start_date) &
            (available_dates <= end_date)
        ]
        
        if self.rebalance_frequency == 'daily':
            return available_dates
        
        elif self.rebalance_frequency == 'weekly':
            # Last trading day of each week
            rebalance_dates = []
            current_week = None
            
            for date in available_dates:
                week = date.isocalendar()[1]
                if week != current_week:
                    if current_week is not None:
                        rebalance_dates.append(last_date)
                    current_week = week
                last_date = date
            
            rebalance_dates.append(last_date)
            return pd.DatetimeIndex(rebalance_dates)
        
        elif self.rebalance_frequency == 'monthly':
            # Last trading day of each month
            rebalance_dates = []
            current_month = None
            
            for date in available_dates:
                month = (date.year, date.month)
                if month != current_month:
                    if current_month is not None:
                        rebalance_dates.append(last_date)
                    current_month = month
                last_date = date
            
            rebalance_dates.append(last_date)
            return pd.DatetimeIndex(rebalance_dates)
        
        else:
            raise ValueError(f"Unknown frequency: {self.rebalance_frequency}")
    
    def _calculate_transaction_cost(
        self,
        turnover: float,
        portfolio_value: float
    ) -> float:
        """
        Calculate transaction costs.
        
        Parameters
        ----------
        turnover : float
            Portfolio turnover (0.0 to 2.0)
        portfolio_value : float
            Current portfolio value
        
        Returns
        -------
        float
            Transaction cost in dollars
        """
        cost_rate = self.transaction_cost_bps / 10000.0  # Convert bps to decimal
        return turnover * portfolio_value * cost_rate
    
    def _calculate_period_returns(
        self,
        weights: Dict[str, float],
        prices: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict[pd.Timestamp, float]:
        """
        Calculate portfolio returns between rebalances.

        Parameters
        ----------
        weights : dict
            Portfolio weights {ticker: weight}
        prices : pd.DataFrame
            Price data
        start_date : pd.Timestamp
            Period start
        end_date : pd.Timestamp
            Period end

        Returns
        -------
        dict
            {date: return} for each day in period
        """
        # Get all dates in period
        all_dates = prices.index.get_level_values('date').unique()
        period_dates = all_dates[
            (all_dates > start_date) &
            (all_dates <= end_date)
        ]

        period_returns = {}

        # Track previous date for daily return calculation
        prev_date = start_date

        for date in period_dates:
            # Calculate weighted return for this single day
            daily_return = 0.0

            for ticker, weight in weights.items():
                try:
                    # Use open price to avoid lookahead bias (signal at t-1 close, execute at t open)
                    price_prev = prices.loc[(prev_date, ticker), 'close']
                    try:
                        price_curr = prices.loc[(date, ticker), 'open']
                    except KeyError:
                        price_curr = prices.loc[(date, ticker), 'close']

                    stock_return = (price_curr - price_prev) / price_prev
                    daily_return += weight * stock_return

                except KeyError:
                    # Missing price data, skip this stock for this day
                    continue

            period_returns[date] = daily_return
            prev_date = date  # Update for next iteration

        return period_returns
    
    def _calculate_metrics(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        
        Returns
        -------
        dict
            Performance metrics
        """
        if len(returns) == 0:
            return {}
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # CAGR
        years = len(returns) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assume 2% risk-free)
        rf_daily = (1.02 ** (1/252)) - 1
        excess_returns = returns - rf_daily
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_observations': len(returns)
        }
