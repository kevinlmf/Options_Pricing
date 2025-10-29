"""
Traditional Quantitative Trading Strategies

Implements classic quantitative trading strategies that can be combined with
options pricing and risk management:
- Mean Reversion
- Momentum/Trend Following
- Statistical Arbitrage
- Volatility Trading
- Delta Hedging

These strategies generate trading signals that are then converted to option
positions using the options pricing models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class TradingSignal:
    """Trading signal from traditional strategy"""
    signal_type: SignalType
    confidence: float  # 0-1
    direction: str  # 'bullish', 'bearish', 'neutral'
    volatility_forecast: Optional[float] = None
    price_target: Optional[float] = None
    strategy_name: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MeanReversionStrategy:
    """
    Mean Reversion Strategy using Bollinger Bands and Z-Score

    Logic:
    - Price > Upper Band (overbought) → Sell signal
    - Price < Lower Band (oversold) → Buy signal
    - Works well in ranging markets with high volatility
    """

    def __init__(self,
                 lookback_period: int = 20,
                 num_std: float = 2.0,
                 zscore_threshold: float = 2.0):
        """
        Parameters:
        -----------
        lookback_period : int
            Period for moving average calculation
        num_std : float
            Number of standard deviations for bands
        zscore_threshold : float
            Z-score threshold for signal generation
        """
        self.lookback_period = lookback_period
        self.num_std = num_std
        self.zscore_threshold = zscore_threshold

    def generate_signal(self,
                       prices: List[float],
                       current_price: float) -> TradingSignal:
        """Generate mean reversion signal"""
        if len(prices) < self.lookback_period:
            return TradingSignal(
                signal_type=SignalType.NEUTRAL,
                confidence=0.0,
                direction='neutral',
                strategy_name='mean_reversion'
            )

        prices_arr = np.array(prices[-self.lookback_period:])

        # Calculate Bollinger Bands
        ma = np.mean(prices_arr)
        std = np.std(prices_arr)
        upper_band = ma + self.num_std * std
        lower_band = ma - self.num_std * std

        # Calculate Z-score
        zscore = (current_price - ma) / std if std > 0 else 0

        # Generate signal
        if zscore > self.zscore_threshold:
            # Overbought - expect reversion down
            signal_type = SignalType.STRONG_SELL if zscore > 3 else SignalType.SELL
            confidence = min(abs(zscore) / 4, 1.0)
            direction = 'bearish'
            price_target = ma
        elif zscore < -self.zscore_threshold:
            # Oversold - expect reversion up
            signal_type = SignalType.STRONG_BUY if zscore < -3 else SignalType.BUY
            confidence = min(abs(zscore) / 4, 1.0)
            direction = 'bullish'
            price_target = ma
        else:
            signal_type = SignalType.NEUTRAL
            confidence = 0.0
            direction = 'neutral'
            price_target = current_price

        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            direction=direction,
            price_target=price_target,
            strategy_name='mean_reversion'
        )


class MomentumStrategy:
    """
    Momentum Strategy using Rate of Change and Moving Average Crossover

    Logic:
    - Strong uptrend → Buy signal (ride the momentum)
    - Strong downtrend → Sell signal
    - Works well in trending markets
    """

    def __init__(self,
                 short_period: int = 10,
                 long_period: int = 50,
                 roc_period: int = 14):
        """
        Parameters:
        -----------
        short_period : int
            Short-term moving average period
        long_period : int
            Long-term moving average period
        roc_period : int
            Rate of change period
        """
        self.short_period = short_period
        self.long_period = long_period
        self.roc_period = roc_period

    def generate_signal(self,
                       prices: List[float],
                       current_price: float) -> TradingSignal:
        """Generate momentum signal"""
        if len(prices) < self.long_period:
            return TradingSignal(
                signal_type=SignalType.NEUTRAL,
                confidence=0.0,
                direction='neutral',
                strategy_name='momentum'
            )

        prices_arr = np.array(prices)

        # Calculate moving averages
        short_ma = np.mean(prices_arr[-self.short_period:])
        long_ma = np.mean(prices_arr[-self.long_period:])

        # Calculate rate of change
        if len(prices) >= self.roc_period:
            roc = (current_price - prices_arr[-self.roc_period]) / prices_arr[-self.roc_period]
        else:
            roc = 0

        # Generate signal based on MA crossover and ROC
        ma_signal = short_ma - long_ma

        # Combine signals
        if ma_signal > 0 and roc > 0.05:
            signal_type = SignalType.STRONG_BUY
            confidence = min(abs(roc) / 0.2, 1.0)
            direction = 'bullish'
        elif ma_signal > 0 and roc > 0:
            signal_type = SignalType.BUY
            confidence = min(abs(roc) / 0.1, 0.7)
            direction = 'bullish'
        elif ma_signal < 0 and roc < -0.05:
            signal_type = SignalType.STRONG_SELL
            confidence = min(abs(roc) / 0.2, 1.0)
            direction = 'bearish'
        elif ma_signal < 0 and roc < 0:
            signal_type = SignalType.SELL
            confidence = min(abs(roc) / 0.1, 0.7)
            direction = 'bearish'
        else:
            signal_type = SignalType.NEUTRAL
            confidence = 0.0
            direction = 'neutral'

        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            direction=direction,
            strategy_name='momentum'
        )


class VolatilityTradingStrategy:
    """
    Volatility Trading Strategy using realized vs implied volatility

    Logic:
    - IV > RV: Volatility is overpriced → Sell volatility (iron condor, short straddle)
    - IV < RV: Volatility is underpriced → Buy volatility (long straddle, strangle)
    """

    def __init__(self,
                 lookback_period: int = 30,
                 iv_rv_threshold: float = 0.2):
        """
        Parameters:
        -----------
        lookback_period : int
            Period for realized volatility calculation
        iv_rv_threshold : float
            Threshold for IV/RV ratio to trigger signal
        """
        self.lookback_period = lookback_period
        self.iv_rv_threshold = iv_rv_threshold

    def calculate_realized_vol(self, prices: List[float]) -> float:
        """Calculate realized volatility from price history"""
        if len(prices) < 2:
            return 0.0

        prices_arr = np.array(prices)
        returns = np.diff(np.log(prices_arr))

        # Annualized realized volatility
        realized_vol = np.std(returns) * np.sqrt(252)
        return realized_vol

    def generate_signal(self,
                       prices: List[float],
                       current_iv: float) -> TradingSignal:
        """Generate volatility trading signal"""
        if len(prices) < self.lookback_period:
            return TradingSignal(
                signal_type=SignalType.NEUTRAL,
                confidence=0.0,
                direction='neutral',
                volatility_forecast=current_iv,
                strategy_name='volatility_trading'
            )

        # Calculate realized volatility
        realized_vol = self.calculate_realized_vol(prices[-self.lookback_period:])

        # Compare IV to RV
        iv_rv_ratio = (current_iv - realized_vol) / realized_vol if realized_vol > 0 else 0

        # Generate signal
        if iv_rv_ratio > self.iv_rv_threshold:
            # IV overpriced - sell volatility
            signal_type = SignalType.STRONG_SELL if iv_rv_ratio > 0.4 else SignalType.SELL
            confidence = min(abs(iv_rv_ratio) / 0.5, 1.0)
            direction = 'sell_volatility'
            vol_forecast = realized_vol
        elif iv_rv_ratio < -self.iv_rv_threshold:
            # IV underpriced - buy volatility
            signal_type = SignalType.STRONG_BUY if iv_rv_ratio < -0.4 else SignalType.BUY
            confidence = min(abs(iv_rv_ratio) / 0.5, 1.0)
            direction = 'buy_volatility'
            vol_forecast = current_iv * (1 + abs(iv_rv_ratio))
        else:
            signal_type = SignalType.NEUTRAL
            confidence = 0.0
            direction = 'neutral'
            vol_forecast = current_iv

        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            direction=direction,
            volatility_forecast=vol_forecast,
            strategy_name='volatility_trading'
        )


class DeltaHedgingStrategy:
    """
    Delta Hedging Strategy to maintain delta neutrality

    Logic:
    - Monitor portfolio delta
    - Generate hedging signals when delta exceeds thresholds
    - Helps maintain market-neutral position
    """

    def __init__(self,
                 delta_threshold: float = 10.0,
                 hedge_ratio: float = 0.5):
        """
        Parameters:
        -----------
        delta_threshold : float
            Delta threshold to trigger hedging
        hedge_ratio : float
            Fraction of delta to hedge (0-1)
        """
        self.delta_threshold = delta_threshold
        self.hedge_ratio = hedge_ratio

    def generate_signal(self,
                       current_delta: float,
                       current_price: float) -> TradingSignal:
        """Generate delta hedging signal"""

        # Determine if hedging is needed
        if abs(current_delta) > self.delta_threshold:
            # Need to hedge
            if current_delta > 0:
                # Long delta - need to sell to hedge
                signal_type = SignalType.SELL
                direction = 'hedge_short'
            else:
                # Short delta - need to buy to hedge
                signal_type = SignalType.BUY
                direction = 'hedge_long'

            confidence = min(abs(current_delta) / (self.delta_threshold * 2), 1.0)
        else:
            signal_type = SignalType.NEUTRAL
            direction = 'neutral'
            confidence = 0.0

        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            direction=direction,
            strategy_name='delta_hedging'
        )


class StrategyEnsemble:
    """
    Ensemble that combines multiple traditional strategies

    Aggregates signals from multiple strategies to produce a consensus signal
    with weighted voting.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Parameters:
        -----------
        weights : dict, optional
            Strategy weights for ensemble voting
            Default: equal weights
        """
        self.mean_reversion = MeanReversionStrategy()
        self.momentum = MomentumStrategy()
        self.volatility = VolatilityTradingStrategy()
        self.delta_hedge = DeltaHedgingStrategy()

        self.weights = weights or {
            'mean_reversion': 0.35,      # Increased from 0.3
            'momentum': 0.35,            # Increased from 0.3
            'volatility_trading': 0.25,  # Same
            'delta_hedging': 0.05        # Reduced from 0.15 to minimize drag
        }

        self.signals_history: List[Dict[str, TradingSignal]] = []

    def generate_ensemble_signal(self,
                                prices: List[float],
                                current_price: float,
                                current_iv: float,
                                current_delta: float = 0.0) -> Dict[str, TradingSignal]:
        """
        Generate signals from all strategies and combine them

        Returns:
        --------
        dict
            Dictionary of signals from each strategy plus ensemble signal
        """
        signals = {}

        # Generate individual signals
        signals['mean_reversion'] = self.mean_reversion.generate_signal(
            prices, current_price
        )

        signals['momentum'] = self.momentum.generate_signal(
            prices, current_price
        )

        signals['volatility_trading'] = self.volatility.generate_signal(
            prices, current_iv
        )

        signals['delta_hedging'] = self.delta_hedge.generate_signal(
            current_delta, current_price
        )

        # Combine signals using weighted voting
        weighted_signal = 0.0
        total_confidence = 0.0

        for strategy_name, signal in signals.items():
            weight = self.weights.get(strategy_name, 0.0)
            weighted_signal += signal.signal_type.value * signal.confidence * weight
            total_confidence += signal.confidence * weight

        # Create ensemble signal (REDUCED THRESHOLD for more trading)
        avg_confidence = total_confidence / sum(self.weights.values())

        if weighted_signal > 0.25:  # Reduced from 0.5 to 0.25
            ensemble_type = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
            ensemble_direction = 'bullish'
        elif weighted_signal < -0.25:  # Reduced from -0.5 to -0.25
            ensemble_type = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
            ensemble_direction = 'bearish'
        else:
            ensemble_type = SignalType.NEUTRAL
            ensemble_direction = 'neutral'

        signals['ensemble'] = TradingSignal(
            signal_type=ensemble_type,
            confidence=avg_confidence,
            direction=ensemble_direction,
            strategy_name='ensemble'
        )

        # Store history
        self.signals_history.append(signals)

        return signals

    def get_signal_summary(self, signals: Dict[str, TradingSignal]) -> str:
        """Get human-readable summary of signals"""
        lines = []
        lines.append("=" * 70)
        lines.append("STRATEGY ENSEMBLE - SIGNAL SUMMARY")
        lines.append("=" * 70)

        for strategy_name, signal in signals.items():
            if strategy_name == 'ensemble':
                lines.append("")
                lines.append("-" * 70)
                lines.append(f"ENSEMBLE SIGNAL: {signal.signal_type.name}")
                lines.append(f"  Direction: {signal.direction.upper()}")
                lines.append(f"  Confidence: {signal.confidence:.1%}")
            else:
                signal_icon = {
                    SignalType.STRONG_BUY: "🟢🟢",
                    SignalType.BUY: "🟢",
                    SignalType.NEUTRAL: "⚪",
                    SignalType.SELL: "🔴",
                    SignalType.STRONG_SELL: "🔴🔴"
                }.get(signal.signal_type, "⚪")

                lines.append(f"\n{strategy_name.upper():>20}: {signal_icon} {signal.signal_type.name}")
                lines.append(f"{'':>20}  Confidence: {signal.confidence:.1%}")

        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("Traditional Strategies Demo\n")

    # Generate sample price data
    np.random.seed(42)
    prices = [45000]
    for _ in range(100):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.03)))

    current_price = prices[-1]
    current_iv = 0.75
    current_delta = 60.0

    # Test ensemble
    ensemble = StrategyEnsemble()
    signals = ensemble.generate_ensemble_signal(
        prices=prices,
        current_price=current_price,
        current_iv=current_iv,
        current_delta=current_delta
    )

    # Print results
    print(ensemble.get_signal_summary(signals))

    print("\nIndividual Strategy Details:")
    print("-" * 70)
    for name, signal in signals.items():
        if name != 'ensemble':
            print(f"\n{name}:")
            print(f"  Signal: {signal.signal_type.name}")
            print(f"  Direction: {signal.direction}")
            print(f"  Confidence: {signal.confidence:.2%}")
            if signal.price_target:
                print(f"  Price Target: ${signal.price_target:,.2f}")
            if signal.volatility_forecast:
                print(f"  Vol Forecast: {signal.volatility_forecast:.1%}")
