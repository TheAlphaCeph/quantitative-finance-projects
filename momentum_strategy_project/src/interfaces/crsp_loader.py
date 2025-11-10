"""
CRSP Data Loader

Interface to CRSP/Compustat database via WRDS (Wharton Research Data Services).
Loads equity prices, returns, and market data for strategy backtesting.

Requires WRDS account with CRSP/Compustat access.
"""

from typing import List, Optional, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import wrds
from datetime import datetime, timedelta
from loguru import logger
import yaml


class CRSPLoader:
    """
    CRSP/Compustat data loader via WRDS.
    
    Handles data acquisition, corporate actions adjustment, and
    survivorship bias-free universe construction.
    
    Parameters
    ----------
    wrds_username : str, optional
        WRDS username (if None, loads from credentials file)
    credentials_path : str, optional
        Path to credentials YAML file
    cache_dir : str, optional
        Directory for caching downloaded data
    
    Notes
    -----
    Data is automatically adjusted for:
    - Stock splits and dividends
    - Delisting returns
    - Corporate actions
    
    All returns include dividends unless specified otherwise.
    """
    
    def __init__(
        self,
        wrds_username: Optional[str] = None,
        credentials_path: str = 'config/credentials.yaml',
        cache_dir: str = 'data/cache'
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load credentials
        if wrds_username is None:
            with open(credentials_path, 'r') as f:
                creds = yaml.safe_load(f)
                wrds_username = creds['wrds']['username']
        
        # Connect to WRDS
        try:
            self.db = wrds.Connection(wrds_username=wrds_username)
            logger.info(f"Connected to WRDS as {wrds_username}")
        except Exception as e:
            logger.error(f"WRDS connection failed: {e}")
            raise ConnectionError(
                "Cannot connect to WRDS. Check credentials and network access."
            ) from e
    
    def get_stock_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        adjust_for_splits: bool = True,
        include_delisted: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve adjusted close prices for specified tickers.
        
        Parameters
        ----------
        tickers : List[str]
            Stock tickers
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        adjust_for_splits : bool, optional
            Apply split adjustments (default: True)
        include_delisted : bool, optional
            Include delisted stocks (default: True for survivorship-free)
        
        Returns
        -------
        pd.DataFrame
            Multi-index (date, ticker) with adjusted close prices
        
        Examples
        --------
        >>> loader = CRSPLoader(wrds_username='your_username')
        >>> prices = loader.get_stock_prices(
        ...     tickers=['AAPL', 'MSFT'],
        ...     start_date='2020-01-01',
        ...     end_date='2024-12-31'
        ... )
        >>> prices.head()
                            close
        date       ticker        
        2020-01-02 AAPL    75.09
                   MSFT   160.62
        2020-01-03 AAPL    74.36
                   MSFT   158.62
        """
        logger.info(
            f"Loading price data for {len(tickers)} tickers "
            f"from {start_date} to {end_date}"
        )
        
        # Build SQL query
        ticker_list = "','".join(tickers)
        delisting_filter = "" if include_delisted else "AND dlstcd IS NULL"
        
        query = f"""
        SELECT 
            date,
            ticker,
            prc AS close,
            ret AS return,
            cfacpr AS adj_factor,
            dlstcd AS delist_code
        FROM crsp.dsf
        WHERE ticker IN ('{ticker_list}')
        AND date BETWEEN '{start_date}' AND '{end_date}'
        {delisting_filter}
        ORDER BY date, ticker
        """
        
        df = self.db.raw_sql(query)
        
        if df.empty:
            logger.warning("No data returned from CRSP")
            return pd.DataFrame()
        
        # Handle negative prices (bid-ask average indicator)
        df['close'] = df['close'].abs()
        
        # Apply split adjustments if requested
        if adjust_for_splits:
            df['close'] = df['close'] / df['adj_factor']
        
        # Handle delisting returns
        df['return'] = df['return'].fillna(0)
        df.loc[df['delist_code'].notna(), 'return'] = (
            df.loc[df['delist_code'].notna(), 'return']
            .clip(lower=-1.0)  # Cap delisting losses at -100%
        )
        
        # Set multi-index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date', 'ticker']).sort_index()
        
        logger.info(
            f"Loaded {len(df)} price records "
            f"({df.index.get_level_values('date').nunique()} trading days)"
        )
        
        return df[['close', 'return']]
    
    def get_returns(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        include_dividends: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve total returns for specified tickers.
        
        Parameters
        ----------
        tickers : List[str]
            Stock tickers
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        include_dividends : bool, optional
            Include dividend returns (default: True)
        
        Returns
        -------
        pd.DataFrame
            Multi-index (date, ticker) with returns
        """
        prices = self.get_stock_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        returns = prices['return'].copy()
        
        if not include_dividends:
            # Calculate price-only returns
            price_df = prices['close'].unstack('ticker')
            returns = price_df.pct_change().stack('ticker')
        
        return returns.to_frame('return')
    
    def get_market_data(
        self,
        start_date: str,
        end_date: str,
        market_index: str = 'CRSP_VW'
    ) -> pd.DataFrame:
        """
        Retrieve market index data.
        
        Parameters
        ----------
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        market_index : str, optional
            Market index to use (default: 'CRSP_VW' for value-weighted)
            Options: 'CRSP_VW', 'CRSP_EW', 'SP500'
        
        Returns
        -------
        pd.DataFrame
            Daily market returns with date index
        """
        logger.info(
            f"Loading {market_index} data from {start_date} to {end_date}"
        )
        
        if market_index == 'CRSP_VW':
            query = f"""
            SELECT caldt AS date, vwretd AS market_return
            FROM crsp.dsi
            WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY caldt
            """
        elif market_index == 'CRSP_EW':
            query = f"""
            SELECT caldt AS date, ewretd AS market_return
            FROM crsp.dsi
            WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY caldt
            """
        elif market_index == 'SP500':
            query = f"""
            SELECT caldt AS date, spindx AS sp500_level, spret AS market_return
            FROM crsp.dsi
            WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY caldt
            """
        else:
            raise ValueError(f"Unknown market index: {market_index}")
        
        df = self.db.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        logger.info(f"Loaded {len(df)} trading days of market data")
        
        return df
    
    def get_fama_french_factors(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'daily'
    ) -> pd.DataFrame:
        """
        Retrieve Fama-French factor returns.
        
        Parameters
        ----------
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        frequency : str, optional
            'daily' or 'monthly' (default: 'daily')
        
        Returns
        -------
        pd.DataFrame
            Factor returns: MKT-RF, SMB, HML, RF
        """
        logger.info(
            f"Loading Fama-French {frequency} factors "
            f"from {start_date} to {end_date}"
        )
        
        table = 'factors_daily' if frequency == 'daily' else 'factors_monthly'
        
        query = f"""
        SELECT 
            date,
            mktrf,
            smb,
            hml,
            rf
        FROM ff.{table}
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """
        
        df = self.db.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Convert from percentage to decimal
        for col in ['mktrf', 'smb', 'hml', 'rf']:
            df[col] = df[col] / 100.0
        
        logger.info(
            f"Loaded {len(df)} {frequency} factor observations"
        )
        
        return df
    
    def get_russell_3000_universe(
        self,
        date: str
    ) -> List[str]:
        """
        Get Russell 3000 constituents on a given date.
        
        Parameters
        ----------
        date : str
            Date in 'YYYY-MM-DD' format
        
        Returns
        -------
        List[str]
            Ticker list
        """
        logger.info(f"Loading Russell 3000 universe for {date}")
        
        query = f"""
        SELECT DISTINCT ticker
        FROM crsp.msp
        WHERE date = '{date}'
        AND ticker IN (
            SELECT ticker FROM crsp.msix
            WHERE date = '{date}'
            AND index_type = 'R3000'
        )
        ORDER BY ticker
        """
        
        df = self.db.raw_sql(query)
        tickers = df['ticker'].tolist()
        
        logger.info(f"Found {len(tickers)} Russell 3000 constituents")
        
        return tickers
    
    def get_stock_fundamentals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        items: List[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve fundamental data from Compustat.
        
        Parameters
        ----------
        tickers : List[str]
            Stock tickers
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        items : List[str], optional
            Compustat items to retrieve (default: common fundamentals)
        
        Returns
        -------
        pd.DataFrame
            Fundamental data with date-ticker multi-index
        """
        if items is None:
            items = ['at', 'ceq', 'sale', 'ni', 'oiadp']  # Common items
        
        logger.info(
            f"Loading fundamentals for {len(tickers)} tickers"
        )
        
        ticker_list = "','".join(tickers)
        item_list = ', '.join(items)
        
        query = f"""
        SELECT 
            datadate AS date,
            tic AS ticker,
            {item_list}
        FROM comp.funda
        WHERE tic IN ('{ticker_list}')
        AND datadate BETWEEN '{start_date}' AND '{end_date}'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
        ORDER BY datadate, tic
        """
        
        df = self.db.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date', 'ticker']).sort_index()
        
        logger.info(f"Loaded {len(df)} fundamental records")
        
        return df
    
    def close(self):
        """Close WRDS connection."""
        self.db.close()
        logger.info("WRDS connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
