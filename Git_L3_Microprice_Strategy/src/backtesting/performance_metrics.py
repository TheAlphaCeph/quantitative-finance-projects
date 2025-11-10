"""
Performance metrics computation and reporting
"""

import pandas as pd
import numpy as np
from typing import Dict


class PerformanceMetrics:
    """Compute comprehensive performance statistics"""

    @staticmethod
    def compute(trades_df: pd.DataFrame) -> Dict:
        """
        Compute performance metrics from backtest trades

        Parameters:
        -----------
        trades_df : pd.DataFrame
            Trade results from BacktestEngine

        Returns:
        --------
        Dict
            Comprehensive performance metrics
        """
        if trades_df.empty:
            return {}

        trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)

        # Calculate cumulative returns
        trades_df['cumulative_net'] = (1 + trades_df['net_return']).cumprod() - 1
        trades_df['cumulative_bh'] = (1 + trades_df['raw_return']).cumprod() - 1

        # Basic statistics
        total_trades = len(trades_df)
        wins = (trades_df['net_return'] > 0).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        avg_return = trades_df['net_return'].mean()
        avg_return_bps = avg_return * 10000

        # Cumulative returns
        total_return = trades_df['cumulative_net'].iloc[-1]
        bh_return = trades_df['cumulative_bh'].iloc[-1]
        excess_return = total_return - bh_return

        # Time-based metrics
        days = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days
        trades_per_day = total_trades / max(days, 1)

        # Annualized Sharpe ratio
        ann_factor = np.sqrt(252 * trades_per_day)
        sharpe = (avg_return / (trades_df['net_return'].std() + 1e-8)) * ann_factor

        # Sortino ratio (downside deviation)
        neg_returns = trades_df['net_return'][trades_df['net_return'] < 0]
        if len(neg_returns) > 0:
            sortino = (avg_return / (neg_returns.std() + 1e-8)) * ann_factor
        else:
            sortino = 0.0

        # Maximum drawdown
        cum = trades_df['cumulative_net'] + 1
        running_max = cum.cummax()
        drawdowns = (cum - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calmar ratio
        if max_drawdown < 0:
            calmar = total_return / abs(max_drawdown)
        else:
            calmar = np.inf

        # Win/loss statistics
        if wins > 0:
            avg_win = trades_df[trades_df['net_return'] > 0]['net_return'].mean() * 10000
        else:
            avg_win = 0.0

        if (total_trades - wins) > 0:
            avg_loss = trades_df[trades_df['net_return'] < 0]['net_return'].mean() * 10000
        else:
            avg_loss = 0.0

        # Profit factor
        gross_profit = trades_df[trades_df['net_return'] > 0]['net_return'].sum()
        gross_loss = abs(trades_df[trades_df['net_return'] < 0]['net_return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return_bps': avg_return_bps,
            'total_return': total_return,
            'buyhold_return': bh_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'avg_win_bps': avg_win,
            'avg_loss_bps': avg_loss,
            'profit_factor': float(profit_factor),
            'trades_per_day': trades_per_day
        }

    @staticmethod
    def print_report(metrics: Dict):
        """
        Print formatted performance report

        Parameters:
        -----------
        metrics : Dict
            Metrics from compute()
        """
        if not metrics:
            print("No metrics to display")
            return

        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        print(f"\n{'TRADING STATISTICS':<30}")
        print(f"  Total Trades:           {metrics['total_trades']:,}")
        print(f"  Win Rate:               {metrics['win_rate']:.2%}")
        print(f"  Trades per Day:         {metrics['trades_per_day']:.1f}")

        print(f"\n{'RETURNS':<30}")
        print(f"  Avg Return/Trade:       {metrics['avg_return_bps']:.2f} bps")
        print(f"  Total Return (Net):     {metrics['total_return']:.2%}")
        print(f"  Buy & Hold Return:      {metrics['buyhold_return']:.2%}")
        print(f"  Excess Return (Alpha):  {metrics['excess_return']:.2%}")

        print(f"\n{'RISK-ADJUSTED PERFORMANCE':<30}")
        print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:          {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:           {metrics['calmar_ratio']:.2f}")

        print(f"\n{'RISK METRICS':<30}")
        print(f"  Maximum Drawdown:       {metrics['max_drawdown']:.2%}")
        print(f"  Profit Factor:          {metrics['profit_factor']:.2f}")
        print(f"  Avg Win:                {metrics['avg_win_bps']:.2f} bps")
        print(f"  Avg Loss:               {metrics['avg_loss_bps']:.2f} bps")

        print("="*70 + "\n")
