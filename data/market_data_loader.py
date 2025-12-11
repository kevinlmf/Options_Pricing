"""
Market Data Loader
==================

Fetches and processes real market data for volatility modeling.
Uses yfinance to retrieve historical data for S&P 500 (SPY) and other assets.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta

class MarketDataLoader:
    """
    Loader for real market data
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        # In a real app, we would implement caching here
        
    def fetch_data(self, 
                  ticker: str = "SPY", 
                  period: str = "2y") -> Tuple[pd.Series, pd.Series]:
        """
        Fetch historical price and return data
        
        Parameters:
        -----------
        ticker : str
            Asset ticker (e.g., 'SPY', '^GSPC')
        period : str
            Data period to download (e.g., '1y', '2y', 'max')
            
        Returns:
        --------
        prices : pd.Series
            Adjusted close prices
        returns : pd.Series
            Daily log returns
        """
        print(f"Fetching data for {ticker} (period={period})...")
        
        try:
            # auto_adjust=True is now default, so 'Close' is adjusted close
            data = yf.download(ticker, period=period, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    prices = data['Close'][ticker]
                except KeyError:
                    # Fallback if ticker level is missing or different
                    prices = data['Close'].iloc[:, 0]
            else:
                prices = data['Close']
            
            # Calculate log returns
            returns = np.log(prices / prices.shift(1)).dropna()
            
            # Align prices to returns
            prices = prices.loc[returns.index]
            
            print(f"Successfully fetched {len(returns)} data points.")
            return prices, returns
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Fallback to synthetic data if fetch fails
            print("Falling back to synthetic data...")
            return self._generate_synthetic_data()
            
    def fetch_market_context(self, ticker: str = "SPY") -> Dict:
        """
        Fetch broader market context for LLM models
        
        Includes:
        - VIX (Volatility Index)
        - Treasury Yields (^TNX)
        - Volume
        """
        try:
            # Fetch VIX
            vix_data = yf.download("^VIX", period="1y", progress=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                current_vix = vix_data['Close'].iloc[-1].item()
            else:
                current_vix = vix_data['Close'].iloc[-1]
            
            # Fetch 10Y Treasury Yield
            tnx_data = yf.download("^TNX", period="1y", progress=False)
            if isinstance(tnx_data.columns, pd.MultiIndex):
                current_rate = tnx_data['Close'].iloc[-1].item() / 100.0
            else:
                current_rate = tnx_data['Close'].iloc[-1] / 100.0
            
            # Fetch Asset Data
            asset = yf.Ticker(ticker)
            info = asset.info
            
            return {
                'vix': current_vix,
                'risk_free_rate': current_rate,
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown')
            }
            
        except Exception as e:
            print(f"Error fetching market context: {e}")
            return {
                'vix': 20.0,
                'risk_free_rate': 0.04,
                'volume': 1000000,
                'market_cap': 1e12,
                'sector': 'Technology'
            }

    def _generate_synthetic_data(self) -> Tuple[pd.Series, pd.Series]:
        """Generate synthetic data as fallback"""
        np.random.seed(42)
        n = 252
        prices = [100.0]
        for _ in range(n):
            ret = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + ret))
            
        price_series = pd.Series(prices)
        returns = price_series.pct_change().dropna()
        return price_series.iloc[1:], returns

if __name__ == "__main__":
    # Test
    loader = MarketDataLoader()
    prices, returns = loader.fetch_data("SPY", start_date="2023-01-01")
    print(f"Last Price: ${prices.iloc[-1]:.2f}")
    print(f"Last Return: {returns.iloc[-1]:.2%}")
    
    context = loader.fetch_market_context("SPY")
    print("Market Context:", context)
