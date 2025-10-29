"""
Bitcoin Options Market Data Fetcher
Fetches real-time and historical Bitcoin options data from Deribit API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeribitDataFetcher:
    """
    Fetches Bitcoin options data from Deribit exchange
    """

    def __init__(self, use_testnet: bool = False):
        """
        Initialize Deribit data fetcher

        Parameters:
        -----------
        use_testnet : bool
            Use testnet API (default: False for production)
        """
        if use_testnet:
            self.base_url = "https://test.deribit.com/api/v2"
        else:
            self.base_url = "https://www.deribit.com/api/v2"

        self.session = requests.Session()

    def get_btc_index_price(self) -> float:
        """
        Get current BTC index price

        Returns:
        --------
        float : Current BTC index price in USD
        """
        try:
            url = f"{self.base_url}/public/get_index_price"
            params = {"index_name": "btc_usd"}
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            return data['result']['index_price']

        except Exception as e:
            logger.error(f"Error fetching BTC index price: {e}")
            raise

    def get_available_instruments(self, currency: str = "BTC",
                                  kind: str = "option",
                                  expired: bool = False) -> List[Dict]:
        """
        Get list of available instruments

        Parameters:
        -----------
        currency : str
            Currency (BTC or ETH)
        kind : str
            Instrument kind: 'future', 'option'
        expired : bool
            Include expired instruments

        Returns:
        --------
        list : List of instrument dictionaries
        """
        try:
            url = f"{self.base_url}/public/get_instruments"
            params = {
                "currency": currency,
                "kind": kind,
                "expired": str(expired).lower()
            }
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            return data['result']

        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            raise

    def get_option_chain(self, currency: str = "BTC",
                        expiration_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get complete option chain for a given expiration

        Parameters:
        -----------
        currency : str
            Currency (BTC or ETH)
        expiration_date : str, optional
            Expiration date in format 'DDMMMYY' (e.g., '31DEC23')
            If None, returns all expirations

        Returns:
        --------
        pd.DataFrame : Option chain data
        """
        instruments = self.get_available_instruments(currency=currency, kind="option")

        if expiration_date:
            instruments = [inst for inst in instruments
                          if expiration_date in inst['instrument_name']]

        option_data = []

        for inst in instruments:
            try:
                # Get book summary for each instrument
                book_data = self.get_order_book(inst['instrument_name'])

                # Parse instrument name: BTC-31DEC23-40000-C
                parts = inst['instrument_name'].split('-')
                strike = float(parts[2])
                option_type = 'call' if parts[3] == 'C' else 'put'
                expiry = parts[1]

                option_data.append({
                    'instrument_name': inst['instrument_name'],
                    'strike': strike,
                    'expiration': expiry,
                    'option_type': option_type,
                    'bid_price': book_data['bid_price'],
                    'ask_price': book_data['ask_price'],
                    'mid_price': book_data['mid_price'],
                    'mark_price': book_data['mark_price'],
                    'mark_iv': book_data['mark_iv'],
                    'bid_iv': book_data.get('bid_iv', np.nan),
                    'ask_iv': book_data.get('ask_iv', np.nan),
                    'volume': book_data['volume'],
                    'open_interest': book_data['open_interest'],
                    'underlying_price': book_data['underlying_price'],
                    'timestamp': book_data['timestamp']
                })

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.warning(f"Error processing {inst['instrument_name']}: {e}")
                continue

        df = pd.DataFrame(option_data)

        # Parse expiration dates
        if not df.empty:
            df['expiration_date'] = pd.to_datetime(df['expiration'], format='%d%b%y')
            df['days_to_expiry'] = (df['expiration_date'] - pd.Timestamp.now()).dt.days
            df['years_to_expiry'] = df['days_to_expiry'] / 365.0

        return df

    def get_order_book(self, instrument_name: str) -> Dict:
        """
        Get order book for a specific instrument

        Parameters:
        -----------
        instrument_name : str
            Instrument name (e.g., 'BTC-31DEC23-40000-C')

        Returns:
        --------
        dict : Order book data including prices, IV, volume, OI
        """
        try:
            url = f"{self.base_url}/public/get_order_book"
            params = {"instrument_name": instrument_name}
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()['result']

            # Calculate mid price
            bid_price = data['best_bid_price'] if data['best_bid_price'] else 0
            ask_price = data['best_ask_price'] if data['best_ask_price'] else 0
            mid_price = (bid_price + ask_price) / 2 if (bid_price and ask_price) else 0

            return {
                'bid_price': bid_price,
                'ask_price': ask_price,
                'mid_price': mid_price,
                'mark_price': data['mark_price'],
                'mark_iv': data['mark_iv'],
                'bid_iv': data.get('bid_iv'),
                'ask_iv': data.get('ask_iv'),
                'volume': data['stats']['volume'],
                'open_interest': data['open_interest'],
                'underlying_price': data['underlying_price'],
                'timestamp': data['timestamp']
            }

        except Exception as e:
            logger.error(f"Error fetching order book for {instrument_name}: {e}")
            raise

    def get_historical_volatility(self, currency: str = "BTC",
                                  period: int = 30) -> Dict:
        """
        Get historical volatility data

        Parameters:
        -----------
        currency : str
            Currency (BTC or ETH)
        period : int
            Period in days

        Returns:
        --------
        dict : Historical volatility metrics
        """
        try:
            url = f"{self.base_url}/public/get_historical_volatility"
            params = {
                "currency": currency
            }
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()['result']

            # Calculate realized volatility from price data
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp')

            # Get recent period
            cutoff = datetime.now() - timedelta(days=period)
            df_recent = df[df['timestamp'] > cutoff]

            # Calculate returns and volatility
            returns = np.diff(np.log(df_recent['price']))
            realized_vol = np.std(returns) * np.sqrt(365)  # Annualized

            return {
                'realized_volatility': realized_vol,
                'period_days': period,
                'data_points': len(df_recent),
                'price_data': df_recent
            }

        except Exception as e:
            logger.warning(f"Historical volatility API not available: {e}")
            # Fallback: estimate from ticker data
            return self._estimate_historical_vol(currency, period)

    def _estimate_historical_vol(self, currency: str, period: int) -> Dict:
        """
        Estimate historical volatility from ticker data
        """
        try:
            # Get ticker data for futures
            url = f"{self.base_url}/public/ticker"
            params = {"instrument_name": f"{currency}-PERPETUAL"}
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()['result']

            # Use mark volatility as proxy
            return {
                'realized_volatility': data.get('mark_iv', 0.8) * 0.9,  # Approximate
                'period_days': period,
                'data_points': 1,
                'note': 'Estimated from current mark IV'
            }

        except Exception as e:
            logger.error(f"Error estimating volatility: {e}")
            return {
                'realized_volatility': 0.8,  # Default assumption
                'period_days': period,
                'data_points': 0,
                'note': 'Default value'
            }

    def get_volatility_surface(self, currency: str = "BTC") -> pd.DataFrame:
        """
        Get implied volatility surface

        Returns:
        --------
        pd.DataFrame : Volatility surface with strikes, maturities, and IVs
        """
        option_chain = self.get_option_chain(currency=currency)

        if option_chain.empty:
            logger.warning("No option data available")
            return pd.DataFrame()

        # Filter for valid IV data
        vol_surface = option_chain[option_chain['mark_iv'].notna()].copy()

        # Add moneyness
        vol_surface['moneyness'] = vol_surface['strike'] / vol_surface['underlying_price']

        # Pivot table for visualization
        surface_pivot = vol_surface.pivot_table(
            index='strike',
            columns='years_to_expiry',
            values='mark_iv',
            aggfunc='mean'
        )

        return vol_surface

    def fetch_market_snapshot(self, currency: str = "BTC") -> Dict:
        """
        Get complete market snapshot for backtesting

        Returns:
        --------
        dict : Complete market data snapshot
        """
        logger.info(f"Fetching market snapshot for {currency}...")

        snapshot = {
            'timestamp': datetime.now(),
            'index_price': self.get_btc_index_price(),
            'option_chain': self.get_option_chain(currency=currency),
            'volatility_surface': self.get_volatility_surface(currency=currency),
            'historical_volatility': self.get_historical_volatility(currency=currency)
        }

        logger.info(f"Snapshot captured: {len(snapshot['option_chain'])} options")

        return snapshot

    def get_historical_option_prices(self, instrument_name: str,
                                    start_timestamp: int,
                                    end_timestamp: int,
                                    resolution: str = "60") -> pd.DataFrame:
        """
        Get historical option prices

        Parameters:
        -----------
        instrument_name : str
            Option instrument name
        start_timestamp : int
            Start timestamp (milliseconds)
        end_timestamp : int
            End timestamp (milliseconds)
        resolution : str
            Time resolution in minutes

        Returns:
        --------
        pd.DataFrame : Historical price data
        """
        try:
            url = f"{self.base_url}/public/get_tradingview_chart_data"
            params = {
                "instrument_name": instrument_name,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "resolution": resolution
            }
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()['result']

            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['ticks'], unit='ms'),
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            })

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()


class MockBitcoinDataGenerator:
    """
    Generate synthetic Bitcoin options data for backtesting when API unavailable
    """

    def __init__(self, spot_price: float = 40000, volatility: float = 0.8):
        """
        Initialize mock data generator

        Parameters:
        -----------
        spot_price : float
            Initial BTC spot price
        volatility : float
            Base volatility level
        """
        self.spot_price = spot_price
        self.volatility = volatility

    def generate_option_chain(self, n_strikes: int = 20,
                             maturities: List[float] = [0.1, 0.25, 0.5, 1.0]) -> pd.DataFrame:
        """
        Generate synthetic option chain

        Parameters:
        -----------
        n_strikes : int
            Number of strike prices
        maturities : list
            List of maturities in years

        Returns:
        --------
        pd.DataFrame : Synthetic option chain
        """
        # Generate strikes around ATM
        strikes = np.linspace(0.8 * self.spot_price, 1.2 * self.spot_price, n_strikes)

        option_data = []

        for T in maturities:
            for K in strikes:
                for option_type in ['call', 'put']:
                    # Generate synthetic implied volatility with smile
                    moneyness = np.log(K / self.spot_price)
                    iv = self._generate_iv_smile(moneyness, T)

                    # Simple Black-Scholes price
                    price = self._bs_price(self.spot_price, K, T, 0.05, iv, option_type)

                    # Add bid-ask spread
                    spread = 0.02 * price
                    bid = price - spread / 2
                    ask = price + spread / 2

                    option_data.append({
                        'strike': K,
                        'expiration': f'{int(T*365)}D',
                        'years_to_expiry': T,
                        'option_type': option_type,
                        'mark_price': price,
                        'bid_price': bid,
                        'ask_price': ask,
                        'mid_price': price,
                        'mark_iv': iv,
                        'underlying_price': self.spot_price,
                        'volume': np.random.randint(10, 1000),
                        'open_interest': np.random.randint(100, 10000),
                        'moneyness': K / self.spot_price,
                        'timestamp': datetime.now()
                    })

        return pd.DataFrame(option_data)

    def _generate_iv_smile(self, log_moneyness: float, T: float) -> float:
        """
        Generate realistic implied volatility smile

        Uses simplified SVI parameterization
        """
        # ATM volatility
        atm_vol = self.volatility

        # Smile parameters
        a = 0.04  # Overall level
        b = 0.3   # Angle of smile
        rho = -0.4  # Skew
        m = 0.0   # ATM location
        sigma_param = 0.2  # Curvature

        # SVI formula
        total_var = a + b * (rho * (log_moneyness - m) +
                            np.sqrt((log_moneyness - m)**2 + sigma_param**2))

        iv = np.sqrt(max(total_var / T, 0.01))

        # Add term structure
        iv *= (1 + 0.1 * np.exp(-2 * T))

        return iv

    def _bs_price(self, S: float, K: float, T: float, r: float,
                 sigma: float, option_type: str) -> float:
        """Black-Scholes pricing"""
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
