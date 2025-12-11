"""
Observable Factor Extractor
============================

Extract observable factors from real market data to explain volatility.

Key Difference from Latent Factors:
- Uses REAL market data (bid-ask spread, volume, etc.)
- Factors are directly observable and verifiable
- More suitable for production trading systems

Observable Factors:
1. Market Maker Factor: Real bid-ask spread → Volatility
2. Arbitrageur Factor: Real trading frequency → Market efficiency
3. Trend Follower Factor: Real momentum indicators → Volatility clustering
4. Fundamental Factor: Real volume/price ratio → Market stability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@dataclass
class ObservableFactors:
    """Observable factors extracted from real market data"""
    market_maker_factor: float      # From bid-ask spread
    arbitrageur_factor: float        # From trading frequency
    trend_follower_factor: float    # From momentum indicators
    fundamental_factor: float        # From volume/price ratio
    timestamp: pd.Timestamp
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'market_maker_vol': self.market_maker_factor,
            'arbitrageur_vol': self.arbitrageur_factor,
            'trend_follower_vol': self.trend_follower_factor,
            'fundamental_vol': self.fundamental_factor
        }


class ObservableFactorExtractor:
    """
    Extract observable factors from real market data.
    
    Factors are extracted from:
    1. Bid-ask spread (if available) or High-Low spread proxy
    2. Trading volume and frequency
    3. Price momentum indicators
    4. Volume-price relationships
    """
    
    def __init__(self, 
                 use_high_low_proxy: bool = True,
                 lookback_window: int = 20):
        """
        Initialize extractor.
        
        Parameters:
        -----------
        use_high_low_proxy : bool
            If True, use (High-Low)/Close as proxy for bid-ask spread
        lookback_window : int
            Window for calculating rolling statistics
        """
        self.use_high_low_proxy = use_high_low_proxy
        self.lookback_window = lookback_window
    
    def extract_market_maker_factor(self, 
                                   market_data: pd.DataFrame,
                                   t: int) -> float:
        """
        Extract Market Maker factor from bid-ask spread.
        
        If bid-ask spread not available, use (High-Low)/Close as proxy.
        
        Logic: Wider spread → Higher uncertainty → Higher volatility
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
            Optionally: ['Bid', 'Ask'] or ['BidAskSpread']
        t : int
            Current time index
            
        Returns:
        --------
        float : Market maker volatility factor (0-1)
        """
        if t < self.lookback_window:
            return 0.15  # Default volatility
        
        # Try to get real bid-ask spread
        if 'BidAskSpread' in market_data.columns:
            spread = market_data.iloc[t]['BidAskSpread']
        elif 'Bid' in market_data.columns and 'Ask' in market_data.columns:
            bid = market_data.iloc[t]['Bid']
            ask = market_data.iloc[t]['Ask']
            spread = (ask - bid) / ((ask + bid) / 2)  # Relative spread
        elif self.use_high_low_proxy:
            # Use (High-Low)/Close as proxy for spread
            high = market_data.iloc[t]['High']
            low = market_data.iloc[t]['Low']
            close = market_data.iloc[t]['Close']
            spread = (high - low) / close
        else:
            # Fallback: use recent volatility as proxy
            returns = market_data['Close'].pct_change().iloc[t-self.lookback_window:t]
            spread = returns.std() * np.sqrt(252) * 0.1  # Scale to spread-like
        
        # Convert spread to volatility factor (0.1 to 0.5)
        # Typical spread: 0.001 (10bps) → vol: 0.15
        # Wide spread: 0.01 (100bps) → vol: 0.35
        volatility_factor = 0.10 + min(spread * 25, 0.40)
        return min(volatility_factor, 1.0)
    
    def extract_arbitrageur_factor(self,
                                  market_data: pd.DataFrame,
                                  t: int) -> float:
        """
        Extract Arbitrageur factor from trading frequency.
        
        Logic: More trading → Faster price discovery → Lower volatility
               Less trading → Slower price discovery → Higher volatility
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with 'Volume' column
        t : int
            Current time index
            
        Returns:
        --------
        float : Arbitrageur volatility factor (0-1)
        """
        if t < self.lookback_window:
            return 0.20  # Default
        
        # Calculate trading frequency from volume
        recent_volumes = market_data['Volume'].iloc[t-self.lookback_window:t]
        current_volume = market_data.iloc[t]['Volume']
        avg_volume = recent_volumes.mean()
        
        if avg_volume == 0:
            return 0.30  # Low liquidity → Higher volatility
        
        # Trading frequency = current_volume / avg_volume
        # High frequency → Lower volatility (efficient market)
        # Low frequency → Higher volatility (inefficient market)
        volume_ratio = current_volume / avg_volume
        
        # Normalize: ratio of 1.0 → vol 0.15, ratio of 0.1 → vol 0.35
        volatility_factor = 0.35 - min(volume_ratio, 2.0) * 0.10
        return max(0.10, min(volatility_factor, 0.50))
    
    def extract_trend_follower_factor(self,
                                     market_data: pd.DataFrame,
                                     t: int) -> float:
        """
        Extract Trend Follower factor from momentum indicators.
        
        Logic: Strong momentum → Herding behavior → Volatility amplification
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with 'Close' column
        t : int
            Current time index
            
        Returns:
        --------
        float : Trend follower volatility factor (0-1)
        """
        if t < self.lookback_window:
            return 0.15  # Default
        
        prices = market_data['Close'].iloc[t-self.lookback_window:t+1]
        
        # Calculate momentum
        short_momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5] if len(prices) >= 5 else 0
        long_momentum = (prices.iloc[-1] - prices.iloc[-self.lookback_window]) / prices.iloc[-self.lookback_window]
        
        # Calculate momentum strength (absolute value)
        momentum_strength = abs(short_momentum) + 0.5 * abs(long_momentum)
        
        # Strong momentum → Herding → Higher volatility
        # Base volatility: 0.15, amplification up to 2x
        base_vol = 0.15
        amplification = 1 + min(momentum_strength * 10, 1.0)
        
        volatility_factor = base_vol * amplification
        return min(volatility_factor, 0.50)
    
    def extract_fundamental_factor(self,
                                  market_data: pd.DataFrame,
                                  t: int) -> float:
        """
        Extract Fundamental factor from volume-price relationship.
        
        Logic: High volume/price ratio → Market stability → Lower volatility
               Low volume/price ratio → Speculation → Higher volatility
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with 'Volume' and 'Close' columns
        t : int
            Current time index
            
        Returns:
        --------
        float : Fundamental volatility factor (0-1)
        """
        if t < self.lookback_window:
            return 0.20  # Default
        
        # Calculate volume-price ratio (VWAP-like indicator)
        recent_volumes = market_data['Volume'].iloc[t-self.lookback_window:t]
        recent_prices = market_data['Close'].iloc[t-self.lookback_window:t]
        
        # Dollar volume = volume * price
        dollar_volumes = recent_volumes * recent_prices
        avg_dollar_volume = dollar_volumes.mean()
        current_dollar_volume = market_data.iloc[t]['Volume'] * market_data.iloc[t]['Close']
        
        if avg_dollar_volume == 0:
            return 0.30
        
        # High dollar volume → Stable market → Lower volatility
        volume_ratio = current_dollar_volume / avg_dollar_volume
        
        # Normalize: ratio > 1 → Lower vol, ratio < 1 → Higher vol
        volatility_factor = 0.30 - min(volume_ratio - 1.0, 1.0) * 0.15
        return max(0.10, min(volatility_factor, 0.40))
    
    def extract_all_factors(self,
                           market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all observable factors from market data.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
            Optionally: ['Bid', 'Ask'] or ['BidAskSpread']
            
        Returns:
        --------
        pd.DataFrame : DataFrame with columns:
            - market_maker_vol
            - arbitrageur_vol
            - trend_follower_vol
            - fundamental_vol
            - combined_volatility (weighted average)
        """
        n = len(market_data)
        factors = []
        
        for t in range(self.lookback_window, n):
            mm_factor = self.extract_market_maker_factor(market_data, t)
            arb_factor = self.extract_arbitrageur_factor(market_data, t)
            tf_factor = self.extract_trend_follower_factor(market_data, t)
            fund_factor = self.extract_fundamental_factor(market_data, t)
            
            # Equal weights (can be optimized)
            combined_vol = (mm_factor + arb_factor + tf_factor + fund_factor) / 4.0
            
            factors.append({
                'market_maker_vol': mm_factor,
                'arbitrageur_vol': arb_factor,
                'trend_follower_vol': tf_factor,
                'fundamental_vol': fund_factor,
                'combined_volatility': combined_vol
            })
        
        # Pad beginning with default values
        default_factors = [factors[0]] * self.lookback_window
        
        result_df = pd.DataFrame(default_factors + factors)
        result_df.index = market_data.index
        
        return result_df


if __name__ == "__main__":
    # Example usage
    from data.market_data import YahooDataProvider
    
    # Load real market data
    provider = YahooDataProvider()
    data = provider.get_price_data('AAPL', '2023-01-01', '2024-01-01')
    
    if not data.empty:
        # Extract observable factors
        extractor = ObservableFactorExtractor()
        factors = extractor.extract_all_factors(data)
        
        print("Observable Factors Extracted:")
        print(factors.head())
        print(f"\nFactor Statistics:")
        print(factors.describe())
    else:
        print("No data available. Using mock data...")
        # Use mock data for demonstration
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        mock_data = pd.DataFrame({
            'Open': 100 + np.random.randn(252).cumsum(),
            'High': 100 + np.random.randn(252).cumsum() + 0.5,
            'Low': 100 + np.random.randn(252).cumsum() - 0.5,
            'Close': 100 + np.random.randn(252).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)
        
        extractor = ObservableFactorExtractor()
        factors = extractor.extract_all_factors(mock_data)
        
        print("Observable Factors from Mock Data:")
        print(factors.head())
        print(f"\nFactor Statistics:")
        print(factors.describe())


