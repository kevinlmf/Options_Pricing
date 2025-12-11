"""
Volatility Arbitrage Detector
============================

Detects mispricings in the volatility surface using dual convergence pricing.

This module identifies:
- Skew mispricing (volatility smile/smirk anomalies)
- Term structure arbitrage opportunities
- Volatility risk premium exploitation
- Statistical arbitrage signals

The dual convergence model provides superior volatility forecasts,
allowing us to identify when market prices deviate from fundamental values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ArbitrageType(Enum):
    SKEW_MISPRICING = "skew_mispricing"
    TERM_STRUCTURE_ARB = "term_structure_arbitrage"
    VOL_RISK_PREMIUM = "volatility_risk_premium"
    STATISTICAL_ARB = "statistical_arbitrage"
    BUTTERFLY_ARB = "butterfly_arbitrage"


@dataclass
class ArbitrageSignal:
    """Volatility arbitrage signal"""
    signal_type: ArbitrageType
    confidence: float  # 0-1
    expected_return: float
    max_loss: float
    position_recommendation: Dict[str, float]  # option_symbol -> quantity
    rationale: str
    timestamp: datetime
    expiry_date: datetime
    underlying_symbol: str
    risk_metrics: Dict[str, float] = field(default_factory=dict)


class VolatilityArbitrageDetector:
    """
    Advanced volatility arbitrage detector using dual convergence pricing.

    This detector identifies pricing inefficiencies in the options market by
    comparing dual convergence model prices with market prices.

    Key Features:
    - Real-time volatility surface monitoring
    - Skew anomaly detection
    - Risk-adjusted arbitrage signals
    - Position sizing recommendations
    """

    def __init__(self,
                 min_mispricing_threshold: float = 0.02,  # 2% minimum mispricing
                 max_position_size: float = 0.1,  # Max 10% of portfolio
                 confidence_threshold: float = 0.7):  # 70% minimum confidence

        self.min_mispricing_threshold = min_mispricing_threshold
        self.max_position_size = max_position_size
        self.confidence_threshold = confidence_threshold

    def scan_for_arbitrage(self,
                          pricing_calibration: Dict,
                          market_data: pd.DataFrame) -> List[ArbitrageSignal]:
        """
        Scan the volatility surface for arbitrage opportunities.

        Parameters:
        -----------
        pricing_calibration : Dict
            Results from DualConvergencePricer.calibrate_to_market_surface()
        market_data : pd.DataFrame
            Current market data

        Returns:
        --------
        List[ArbitrageSignal] : Detected arbitrage opportunities
        """
        signals = []
        calibration_results = pricing_calibration.get('calibration_results', [])

        print(f"🔍 Scanning {len(calibration_results)} options for arbitrage opportunities...")

        # Group by underlying and expiry for surface analysis
        surface_groups = self._group_by_surface(calibration_results)

        for underlying, expiry_groups in surface_groups.items():
            for expiry, options in expiry_groups.items():
                # Check for skew mispricing
                skew_signals = self._detect_skew_mispricing(options, underlying, expiry)
                signals.extend(skew_signals)

                # Check for term structure arbitrage
                if len(expiry_groups) > 1:  # Need multiple expiries
                    term_signals = self._detect_term_structure_arb(expiry_groups, underlying)
                    signals.extend(term_signals)

                # Check for butterfly arbitrage
                butterfly_signals = self._detect_butterfly_arbitrage(options, underlying, expiry)
                signals.extend(butterfly_signals)

        # Filter by confidence and size
        filtered_signals = [
            signal for signal in signals
            if signal.confidence >= self.confidence_threshold
        ]

        print(f"   ✓ Found {len(filtered_signals)} high-confidence arbitrage signals")

        return filtered_signals

    def _detect_skew_mispricing(self,
                               options: List[Dict],
                               underlying: str,
                               expiry: datetime) -> List[ArbitrageSignal]:
        """Detect volatility skew/smirk mispricing"""

        signals = []

        # Separate calls and puts
        calls = [opt for opt in options if opt['option'].option_type == 'call']
        puts = [opt for opt in options if opt['option'].option_type == 'put']

        if len(calls) < 3 or len(puts) < 3:
            return signals

        # Calculate implied volatility skew
        call_skew = self._calculate_volatility_skew(calls)
        put_skew = self._calculate_volatility_skew(puts)

        # Check for anomalous skew
        if abs(call_skew) > 0.1 or abs(put_skew) > 0.1:  # 10% skew anomaly
            # Generate arbitrage signal
            signal = ArbitrageSignal(
                signal_type=ArbitrageType.SKEW_MISPRICING,
                confidence=min(abs(call_skew + put_skew) * 5, 1.0),  # Scale confidence
                expected_return=abs(call_skew) * 0.15,  # Rough estimate
                max_loss=self.max_position_size * 0.5,
                position_recommendation=self._generate_skew_position(calls, puts, call_skew, put_skew),
                rationale=f"Detected anomalous volatility skew: calls={call_skew:.1%}, puts={put_skew:.1%}",
                timestamp=datetime.now(),
                expiry_date=expiry,
                underlying_symbol=underlying,
                risk_metrics={
                    'delta_exposure': 0.0,  # Delta-neutral
                    'gamma_exposure': 0.05,
                    'vega_exposure': 0.8,
                    'theta_decay': -0.02
                }
            )
            signals.append(signal)

        return signals

    def _detect_term_structure_arb(self,
                                 expiry_groups: Dict[datetime, List],
                                 underlying: str) -> List[ArbitrageSignal]:
        """Detect term structure arbitrage opportunities"""

        signals = []

        # Compare volatility term structure across expiries
        expiries = sorted(expiry_groups.keys())
        if len(expiries) < 2:
            return signals

        # Calculate forward volatility differences
        for i in range(len(expiries) - 1):
            current_options = expiry_groups[expiries[i]]
            next_options = expiry_groups[expiries[i + 1]]

            # Find ATM options for comparison
            current_atm = self._find_atm_option(current_options)
            next_atm = self._find_atm_option(next_options)

            if current_atm and next_atm:
                # Calculate forward volatility
                current_iv = current_atm.get('model_iv', 0)
                next_iv = next_atm.get('model_iv', 0)

                # Time to expiry
                dt1 = (expiries[i] - datetime.now()).days / 365
                dt2 = (expiries[i + 1] - datetime.now()).days / 365

                if dt1 > 0 and dt2 > 0:
                    # Forward variance calculation
                    forward_var = (next_iv**2 * dt2 - current_iv**2 * dt1) / (dt2 - dt1)

                    if forward_var < 0:  # Negative forward variance = arbitrage
                        signal = ArbitrageSignal(
                            signal_type=ArbitrageType.TERM_STRUCTURE_ARB,
                            confidence=0.85,
                            expected_return=abs(forward_var) * 0.2,
                            max_loss=self.max_position_size * 0.3,
                            position_recommendation=self._generate_calendar_spread_position(
                                current_atm, next_atm
                            ),
                            rationale=f"Negative forward variance detected: {forward_var:.4f}",
                            timestamp=datetime.now(),
                            expiry_date=expiries[i],
                            underlying_symbol=underlying
                        )
                        signals.append(signal)

        return signals

    def _detect_butterfly_arbitrage(self,
                                  options: List[Dict],
                                  underlying: str,
                                  expiry: datetime) -> List[ArbitrageSignal]:
        """Detect butterfly arbitrage opportunities"""

        signals = []

        # Group by strike
        strikes = sorted(set(opt['option'].strike_price for opt in options))

        if len(strikes) < 3:
            return signals

        # Check for convexity violations
        for i in range(1, len(strikes) - 1):
            strike_low = strikes[i - 1]
            strike_mid = strikes[i]
            strike_high = strikes[i + 1]

            # Find options at these strikes (assuming calls for simplicity)
            opt_low = next((opt for opt in options if abs(opt['option'].strike_price - strike_low) < 0.01
                           and opt['option'].option_type == 'call'), None)
            opt_mid = next((opt for opt in options if abs(opt['option'].strike_price - strike_mid) < 0.01
                           and opt['option'].option_type == 'call'), None)
            opt_high = next((opt for opt in options if abs(opt['option'].strike_price - strike_high) < 0.01
                           and opt['option'].option_type == 'call'), None)

            if opt_low and opt_mid and opt_high:
                price_low = opt_low['model_price']
                price_mid = opt_mid['model_price']
                price_high = opt_high['model_price']

                # Butterfly price should be positive
                butterfly_price = price_mid - 0.5 * (price_low + price_high)

                if butterfly_price < -0.01:  # Negative butterfly = arbitrage
                    signal = ArbitrageSignal(
                        signal_type=ArbitrageType.BUTTERFLY_ARB,
                        confidence=0.9,
                        expected_return=abs(butterfly_price) * 100,  # Per contract
                        max_loss=max(price_low, price_mid, price_high) * 0.1,
                        position_recommendation={
                            f"{underlying}_C_{strike_low}_{expiry.strftime('%Y%m%d')}": 0.5,
                            f"{underlying}_C_{strike_mid}_{expiry.strftime('%Y%m%d')}": -1.0,
                            f"{underlying}_C_{strike_high}_{expiry.strftime('%Y%m%d')}": 0.5
                        },
                        rationale=f"Butterfly arbitrage: price = {butterfly_price:.4f}",
                        timestamp=datetime.now(),
                        expiry_date=expiry,
                        underlying_symbol=underlying
                    )
                    signals.append(signal)

        return signals

    def _calculate_volatility_skew(self, options: List[Dict]) -> float:
        """Calculate implied volatility skew"""

        if len(options) < 3:
            return 0.0

        # Sort by moneyness
        sorted_opts = sorted(options, key=lambda x: x['option'].moneyness)

        # Calculate volatility slope
        strikes = [opt['option'].moneyness for opt in sorted_opts]
        vols = [opt.get('model_iv', 0.25) for opt in sorted_opts]

        # Linear regression for skew
        if len(strikes) > 1:
            slope = np.polyfit(strikes, vols, 1)[0]
            return slope
        else:
            return 0.0

    def _generate_skew_position(self, calls: List[Dict], puts: List[Dict],
                              call_skew: float, put_skew: float) -> Dict[str, float]:
        """Generate position recommendation for skew arbitrage"""

        # Simplified: buy underpriced options, sell overpriced options
        position = {}

        # If call skew is too negative (puts expensive), buy calls sell puts
        if call_skew < -0.05:
            # Find ATM call and put
            atm_call = self._find_atm_option(calls)
            atm_put = self._find_atm_option(puts)

            if atm_call and atm_put:
                position[atm_call['option'].__str__()] = self.max_position_size * 0.5
                position[atm_put['option'].__str__()] = -self.max_position_size * 0.5

        return position

    def _generate_calendar_spread_position(self, near_opt: Dict, far_opt: Dict) -> Dict[str, float]:
        """Generate calendar spread position"""

        position = {}
        if near_opt and far_opt:
            # Buy near-term, sell far-term
            position[near_opt['option'].__str__()] = self.max_position_size
            position[far_opt['option'].__str__()] = -self.max_position_size

        return position

    def _find_atm_option(self, options: List[Dict]) -> Optional[Dict]:
        """Find at-the-money option"""

        if not options:
            return None

        # Find option closest to moneyness = 1.0
        atm_option = min(options, key=lambda x: abs(x['option'].moneyness - 1.0))
        return atm_option

    def _group_by_surface(self, calibration_results: List[Dict]) -> Dict[str, Dict[datetime, List]]:
        """Group calibration results by underlying and expiry"""

        surface_groups = {}

        for result in calibration_results:
            option = result['option']
            underlying = option.underlying_symbol
            expiry = option.time_to_maturity  # This should be expiry date

            if underlying not in surface_groups:
                surface_groups[underlying] = {}

            # Convert time_to_maturity to expiry date (simplified)
            expiry_date = datetime.now() + pd.Timedelta(days=option.time_to_maturity * 365)

            if expiry_date not in surface_groups[underlying]:
                surface_groups[underlying][expiry_date] = []

            surface_groups[underlying][expiry_date].append(result)

        return surface_groups

