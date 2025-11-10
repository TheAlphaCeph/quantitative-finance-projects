"""
Sentiment API Interface
The sentiment layer is separate infrastructure; this module only consumes the API.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import requests
from datetime import datetime
from loguru import logger


class SentimentAPI:
    """
    Interface to NLP Sentiment Platform.
    
    Retrieves sentiment scores and metadata for equity earnings transcripts.
    Authentication required via API key.
    
    Parameters
    ----------
    credentials_path : str or Path
        Path to YAML file containing API credentials
    base_url : str, optional
        API endpoint URL (defaults to production)
    timeout : int, optional
        Request timeout in seconds (default: 30)
    max_retries : int, optional
        Maximum retry attempts for failed requests (default: 3)
    
    Notes
    -----
    Sentiment scores are normalized to [-1, 1] range:
    - Positive: Bullish sentiment
    - Negative: Bearish sentiment  
    - Magnitude: Confidence level
    
    API returns daily aggregated scores from all available transcripts
    for each ticker. Scores include confidence intervals and metadata.
    """
    
    def __init__(
        self,
        credentials_path: str = 'config/credentials.yaml',
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.credentials_path = Path(credentials_path)
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Load credentials
        with open(self.credentials_path, 'r') as f:
            creds = yaml.safe_load(f)
            self.api_key = creds['sentiment_api']['api_key']
            self.base_url = base_url or creds['sentiment_api']['base_url']
        
        # Validate connection
        self._validate_connection()
        
        logger.info(
            f"SentimentAPI initialized with endpoint: {self.base_url}"
        )
    
    def _validate_connection(self) -> None:
        """Test API connection and authentication."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("API connection validated successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"API connection failed: {e}")
            raise ConnectionError(
                f"Cannot connect to sentiment API at {self.base_url}. "
                f"Check credentials and network access."
            ) from e
    
    def get_sentiment_history(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        include_metadata: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve historical sentiment scores for specified tickers.
        
        Parameters
        ----------
        tickers : List[str]
            Stock tickers (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        include_metadata : bool, optional
            Include confidence intervals and metadata (default: False)
        
        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame (date, ticker) with sentiment scores
            Columns: 'sentiment' (core score), plus optional metadata
        
        Examples
        --------
        >>> api = SentimentAPI()
        >>> sentiment = api.get_sentiment_history(
        ...     tickers=['AAPL', 'MSFT'],
        ...     start_date='2020-01-01',
        ...     end_date='2024-12-31'
        ... )
        >>> sentiment.head()
                            sentiment
        date       ticker            
        2020-01-02 AAPL       0.23
                   MSFT       0.15
        2020-01-03 AAPL       0.21
                   MSFT       0.18
        """
        logger.info(
            f"Fetching sentiment for {len(tickers)} tickers "
            f"from {start_date} to {end_date}"
        )
        
        # Prepare API request
        endpoint = f"{self.base_url}/sentiment/history"
        params = {
            'tickers': ','.join(tickers),
            'start_date': start_date,
            'end_date': end_date,
            'include_metadata': str(include_metadata).lower()
        }
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    endpoint,
                    params=params,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"API request failed after {self.max_retries} attempts: {e}"
                    )
                    raise
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying...")
        
        # Parse response
        data = response.json()
        df = pd.DataFrame(data['sentiment_scores'])
        
        # Convert to proper datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set multi-index
        df = df.set_index(['date', 'ticker']).sort_index()
        
        logger.info(
            f"Retrieved {len(df)} sentiment records "
            f"({df.index.get_level_values('date').nunique()} trading days)"
        )
        
        return df
    
    def get_latest_sentiment(
        self,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Get most recent sentiment scores for tickers.
        
        Parameters
        ----------
        tickers : List[str]
            Stock tickers
        
        Returns
        -------
        pd.DataFrame
            Latest sentiment with timestamp
        """
        logger.info(f"Fetching latest sentiment for {len(tickers)} tickers")
        
        endpoint = f"{self.base_url}/sentiment/latest"
        params = {'tickers': ','.join(tickers)}
        
        response = requests.get(
            endpoint,
            params=params,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data['latest_sentiment'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df.set_index('ticker')
    
    def get_transcript_metadata(
        self,
        ticker: str,
        fiscal_quarter: str
    ) -> Dict[str, Any]:
        """
        Retrieve metadata for specific earnings transcript.
        
        Parameters
        ----------
        ticker : str
            Stock ticker
        fiscal_quarter : str
            Format: 'YYYY-Q#' (e.g., '2024-Q3')
        
        Returns
        -------
        dict
            Transcript metadata including word count, participants,
            sentiment breakdown, and quality metrics
        """
        logger.info(
            f"Fetching transcript metadata for {ticker} {fiscal_quarter}"
        )
        
        endpoint = f"{self.base_url}/transcripts/metadata"
        params = {
            'ticker': ticker,
            'fiscal_quarter': fiscal_quarter
        }
        
        response = requests.get(
            endpoint,
            params=params,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def batch_download(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        output_path: str
    ) -> None:
        """
        Download bulk sentiment data to local parquet file.
        
        Useful for backtesting with large universes. Handles pagination
        and rate limiting automatically.
        
        Parameters
        ----------
        tickers : List[str]
            Stock tickers
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        output_path : str
            Path for output parquet file
        """
        logger.info(
            f"Starting batch download for {len(tickers)} tickers "
            f"to {output_path}"
        )
        
        # Use pagination for large requests
        batch_size = 100
        all_data = []
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            logger.info(
                f"Downloading batch {i//batch_size + 1}/"
                f"{(len(tickers)-1)//batch_size + 1}"
            )
            
            batch_data = self.get_sentiment_history(
                tickers=batch_tickers,
                start_date=start_date,
                end_date=end_date,
                include_metadata=True
            )
            all_data.append(batch_data)
        
        # Combine and save
        full_data = pd.concat(all_data, axis=0)
        full_data.to_parquet(output_path, compression='snappy')
        
        logger.info(
            f"Batch download complete: {len(full_data)} records "
            f"saved to {output_path}"
        )
    
    def get_universe_coverage(
        self,
        date: str
    ) -> List[str]:
        """
        Get list of all tickers with sentiment coverage on a given date.
        
        Parameters
        ----------
        date : str
            Date in 'YYYY-MM-DD' format
        
        Returns
        -------
        List[str]
            Tickers with sentiment data available
        """
        logger.info(f"Fetching universe coverage for {date}")
        
        endpoint = f"{self.base_url}/universe/coverage"
        params = {'date': date}
        
        response = requests.get(
            endpoint,
            params=params,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        return data['tickers']
