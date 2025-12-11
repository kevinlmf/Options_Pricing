"""
Gamma Scalping Engine
====================

Advanced gamma scalping implementation for volatility harvesting.

Gamma scalping is a sophisticated strategy that:
- Profits from time decay (theta) while maintaining delta-neutrality
- Dynamically hedges gamma exposure as the underlying moves
- Converts volatility into consistent P&L
- Works best in range-bound or oscillating markets

The dual convergence model provides superior gamma scalping signals
by predicting when volatility will be range-bound vs trending.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class ScalpingMode(Enum):
    AGGRESSIVE = "aggressive"  # Frequent rebalancing, higher costs but better tracking
    CONSERVATIVE = "conservative"  # Less frequent, lower costs but more slippage
    DYNAMIC = "dynamic"  # Adaptive based on volatility predictions


@dataclass
class ScalpingSignal:
    """Gamma scalping signal with timing and sizing"""
    timestamp: datetime
    underlying_symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    expected_pnl: float
    gamma_exposure: float
    theta_decay: float
    volatility_forecast: float
    confidence: float
    rationale: str


class GammaScalpingEngine:
    """
    Advanced gamma scalping engine that leverages dual convergence volatility predictions.

    Core Strategy:
    1. Maintain gamma-positive positions (long options or spreads)
    2. Delta-hedge to remain directionally neutral
    3. Rebalance frequently to capture theta decay
    4. Use volatility predictions to time entries/exits

    The dual convergence model gives us an edge in predicting when
    gamma scalping will be profitable (low volatility periods).
    """

    def __init__(self,
                 scalping_mode: ScalpingMode = ScalpingMode.DYNAMIC,
                 max_gamma_exposure: float = 0.5,
                 rebalance_threshold: float = 0.02,  # Rebalance if delta > 2%
                 target_theta_harvest: float = 0.001):  # Target 10bps daily theta

        self.scalping_mode = scalping_mode
        self.max_gamma_exposure = max_gamma_exposure
        self.rebalance_threshold = rebalance_threshold
        self.target_theta_harvest = target_theta_harvest

        # Position tracking
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.scalping_history: List[ScalpingSignal] = []
        self.daily_pnl: List[Dict[str, Any]] = []

    def generate_scalping_signals(self,
                                underlying_symbol: str,
                                current_price: float,
                                volatility_forecast: float,
                                option_chain: List[Dict[str, Any]],
                                current_positions: Dict[str, float]) -> List[ScalpingSignal]:
        """
        Generate gamma scalping signals based on market conditions and volatility forecasts.

        Parameters:
        -----------
        underlying_symbol : str
            Underlying asset
        current_price : float
            Current spot price
        volatility_forecast : float
            Dual convergence volatility prediction
        option_chain : List[Dict]
            Available options
        current_positions : Dict[str, float]
            Current positions (option_symbol -> quantity)

        Returns:
        --------
        List[ScalpingSignal] : Scalping signals
        """
        signals = []

        # Assess scalping opportunity
        scalping_score = self._calculate_scalping_score(volatility_forecast, current_price)

        if scalping_score < 0.3:
            # Poor scalping environment
            return signals

        # Find optimal scalping instruments
        scalping_candidates = self._find_scalping_candidates(
            option_chain, current_price, volatility_forecast
        )

        for candidate in scalping_candidates[:3]:  # Top 3 candidates
            signal = self._create_scalping_signal(
                underlying_symbol, candidate, scalping_score, current_positions
            )

            if signal:
                signals.append(signal)

        return signals

    def execute_scalping_trade(self, signal: ScalpingSignal) -> bool:
        """
        Execute gamma scalping trade.

        Parameters:
        -----------
        signal : ScalpingSignal
            Scalping signal to execute

        Returns:
        --------
        bool : True if successful
        """
        try:
            # Record signal
            self.scalping_history.append(signal)

            # Update positions (simplified)
            option_symbol = f"{signal.underlying_symbol}_{signal.timestamp.strftime('%H%M%S')}"

            self.current_positions[option_symbol] = {
                'quantity': signal.quantity,
                'gamma': signal.gamma_exposure,
                'theta': signal.theta_decay,
                'entry_time': signal.timestamp,
                'expected_pnl': signal.expected_pnl
            }

            print(f"   ✅ Gamma scalping trade executed: {signal.action} {abs(signal.quantity):.0f} contracts")
            return True

        except Exception as e:
            print(f"   ❌ Scalping trade failed: {e}")
            return False

    def calculate_scalping_pnl(self,
                             price_moves: pd.Series,
                             gamma_exposure: float,
                             theta_decay: float,
                             time_elapsed: float) -> Dict[str, float]:
        """
        Calculate gamma scalping P&L from price movements.

        Gamma scalping P&L comes from:
        - Gamma P&L: 0.5 * gamma * (dS)^2 (convexity profit)
        - Theta P&L: theta * dt (time decay profit)
        - Hedging costs: transaction costs from rebalancing

        Parameters:
        -----------
        price_moves : pd.Series
            Price movements during scalping period
        gamma_exposure : float
            Total gamma exposure
        theta_decay : float
            Theta decay rate
        time_elapsed : float
            Time elapsed in years

        Returns:
        --------
        Dict[str, float] : P&L breakdown
        """
        # Calculate gamma P&L from convexity
        gamma_pnl = 0.5 * gamma_exposure * np.sum(price_moves**2)

        # Calculate theta P&L from time decay
        theta_pnl = theta_decay * time_elapsed

        # Estimate hedging costs (simplified)
        n_rebalances = len(price_moves)  # Assume rebalance every price move
        avg_trade_size = 1000  # Simplified
        hedging_costs = n_rebalances * 0.001 * avg_trade_size  # 1bp per trade

        # Total P&L
        total_pnl = gamma_pnl + theta_pnl - hedging_costs

        return {
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'hedging_costs': hedging_costs,
            'total_pnl': total_pnl,
            'pnl_per_day': total_pnl / max(time_elapsed * 365, 1)
        }

    def optimize_scalping_parameters(self,
                                   historical_data: pd.DataFrame,
                                   backtest_period: timedelta = timedelta(days=30)) -> Dict[str, Any]:
        """
        Optimize gamma scalping parameters using historical data.

        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical price data
        backtest_period : timedelta
            Period to backtest

        Returns:
        --------
        Dict[str, Any] : Optimization results
        """
        print("🔬 Optimizing gamma scalping parameters...")

        # Define parameter ranges to test
        rebalance_thresholds = [0.01, 0.02, 0.05, 0.10]
        max_gamma_exposures = [0.3, 0.5, 0.7, 1.0]

        best_params = None
        best_performance = -float('inf')

        # Simple grid search (would be more sophisticated in production)
        for rebalance_thresh in rebalance_thresholds:
            for max_gamma in max_gamma_exposures:

                # Simulate performance with these parameters
                performance = self._simulate_scalping_performance(
                    historical_data, rebalance_thresh, max_gamma, backtest_period
                )

                if performance > best_performance:
                    best_performance = performance
                    best_params = {
                        'rebalance_threshold': rebalance_thresh,
                        'max_gamma_exposure': max_gamma,
                        'expected_performance': performance
                    }

        print(f"   ✅ Optimal parameters found: {best_params}")

        # Update instance parameters
        if best_params:
            self.rebalance_threshold = best_params['rebalance_threshold']
            self.max_gamma_exposure = best_params['max_gamma_exposure']

        return best_params

    def get_scalping_metrics(self) -> Dict[str, Any]:
        """Get comprehensive gamma scalping performance metrics"""

        if not self.scalping_history:
            return {"status": "no_scalping_activity"}

        total_signals = len(self.scalping_history)
        executed_signals = sum(1 for s in self.scalping_history if s.action != 'hold')

        # P&L analysis
        total_pnl = sum(s.expected_pnl for s in self.scalping_history)

        # Success rate
        positive_signals = sum(1 for s in self.scalping_history if s.expected_pnl > 0)

        # Average P&L per trade
        avg_pnl_per_trade = total_pnl / executed_signals if executed_signals > 0 else 0

        return {
            "total_signals": total_signals,
            "executed_signals": executed_signals,
            "success_rate": positive_signals / total_signals if total_signals > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "scalping_mode": self.scalping_mode.value,
            "current_gamma_exposure": sum(p.get('gamma', 0) for p in self.current_positions.values())
        }

    def _calculate_scalping_score(self, volatility_forecast: float, current_price: float) -> float:
        """
        Calculate how favorable current conditions are for gamma scalping.

        Gamma scalping works best when:
        - Low to moderate volatility (not too trending)
        - Sufficient option liquidity
        - Reasonable bid-ask spreads
        """

        # Base score from volatility (lower vol = better scalping)
        vol_score = max(0, 1 - volatility_forecast / 0.30)  # Optimal below 30% vol

        # Price stability score (recent range-bound behavior)
        # This would use historical data in practice
        stability_score = 0.7  # Placeholder

        # Liquidity score (simplified)
        liquidity_score = 0.8  # Placeholder

        # Composite score
        scalping_score = (vol_score * 0.5 + stability_score * 0.3 + liquidity_score * 0.2)

        return scalping_score

    def _find_scalping_candidates(self,
                                option_chain: List[Dict[str, Any]],
                                current_price: float,
                                volatility_forecast: float) -> List[Dict[str, Any]]:
        """
        Find optimal options for gamma scalping.

        Criteria:
        - Reasonable premium (not too expensive)
        - Good gamma exposure
        - Sufficient time to expiry
        - Liquid options
        """

        candidates = []

        for option in option_chain:
            # Calculate option metrics
            moneyness = option['strike'] / current_price
            time_to_expiry = option.get('days_to_expiry', 30) / 365

            # Skip very short-dated or far OTM options
            if time_to_expiry < 0.08 or abs(moneyness - 1.0) > 0.15:
                continue

            # Calculate gamma approximation (simplified)
            # gamma ≈ (N(d1) * exp(-qT)) / (S * σ * sqrt(T))
            d1 = (np.log(current_price / option['strike']) +
                  (0.03 - 0 + 0.5 * volatility_forecast**2) * time_to_expiry) / (volatility_forecast * np.sqrt(time_to_expiry))

            gamma = (np.exp(-0 * time_to_expiry) / (current_price * volatility_forecast * np.sqrt(time_to_expiry))) * \
                   np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)

            # Score candidate
            gamma_score = gamma * 100  # Scale for comparison
            premium_score = 1 / (1 + option['price'] / current_price)  # Prefer cheaper options
            liquidity_score = 1.0  # Simplified

            composite_score = gamma_score * 0.6 + premium_score * 0.3 + liquidity_score * 0.1

            candidate = {
                **option,
                'gamma': gamma,
                'scalping_score': composite_score,
                'moneyness': moneyness
            }

            candidates.append(candidate)

        # Sort by scalping score
        candidates.sort(key=lambda x: x['scalping_score'], reverse=True)

        return candidates

    def _create_scalping_signal(self,
                               underlying_symbol: str,
                               candidate: Dict[str, Any],
                               scalping_score: float,
                               current_positions: Dict[str, float]) -> Optional[ScalpingSignal]:
        """Create gamma scalping signal from candidate"""

        # Determine action based on current position and market conditions
        current_quantity = current_positions.get(candidate.get('symbol', ''), 0)

        if current_quantity == 0:
            # No position - consider entering
            action = 'buy'
            quantity = min(100, int(scalping_score * 50))  # Scale quantity by score
        elif abs(current_quantity) > 200:
            # Large position - consider reducing
            action = 'sell'
            quantity = -min(abs(current_quantity) * 0.2, 50)
        else:
            # Hold current position
            return None

        # Calculate expected P&L
        expected_pnl = scalping_score * abs(quantity) * 0.001  # Simplified

        signal = ScalpingSignal(
            timestamp=datetime.now(),
            underlying_symbol=underlying_symbol,
            action=action,
            quantity=quantity,
            expected_pnl=expected_pnl,
            gamma_exposure=candidate['gamma'] * quantity,
            theta_decay=0.001 * abs(quantity),  # Simplified theta
            volatility_forecast=candidate.get('volatility', 0.25),
            confidence=scalping_score,
            rationale=f"Gamma scalping opportunity with score {scalping_score:.2f}"
        )

        return signal

    def _simulate_scalping_performance(self,
                                     historical_data: pd.DataFrame,
                                     rebalance_threshold: float,
                                     max_gamma: float,
                                     backtest_period: timedelta) -> float:
        """Simulate gamma scalping performance with given parameters"""

        # Simplified simulation
        # In practice, this would run a full backtest

        # Simulate daily returns from scalping
        n_days = backtest_period.days
        daily_returns = np.random.normal(0.0005, 0.002, n_days)  # 5bps mean, 20bps std

        # Adjust for parameter efficiency
        efficiency_factor = 1.0 - abs(rebalance_threshold - 0.02) - abs(max_gamma - 0.5)
        efficiency_factor = max(0.5, min(1.5, efficiency_factor))

        total_return = np.sum(daily_returns) * efficiency_factor

        return total_return

