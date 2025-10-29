"""
Multi-Agent Structural Forecaster

Structural Approach: Simulate agent behaviors → Infer market parameters
vs. Reduced-form: Fit statistical models directly to data

Key Innovation:
- Interpretable: Parameters come from behavioral mechanisms
- Robust: Less prone to spurious correlations
- Adaptive: Agent behaviors can adjust to regime changes
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

from .agents import MarketMaker, Arbitrageur, NoiseTrader, simulate_agent_interaction

logger = logging.getLogger(__name__)


class MultiAgentForecaster:
    """
    Multi-Agent Structural Model for Parameter Forecasting

    Architecture:
    1. Initialize agents with behavioral rules
    2. Simulate agent interactions on historical data
    3. Infer parameters from agent actions:
       - MarketMaker → implied_volatility
       - Arbitrageur → implied_drift
       - NoiseTrader → regime

    Output: (σ_implied, μ_implied, regime)
    """

    def __init__(self,
                 market_maker_risk_aversion: float = 1.0,
                 arbitrageur_risk_aversion: float = 0.5,
                 noise_trader_risk_aversion: float = 2.0,
                 simulation_periods: int = 100):
        """
        Initialize multi-agent forecaster.

        Parameters:
        -----------
        *_risk_aversion : float
            Risk aversion parameters for each agent type
        simulation_periods : int
            Number of periods to simulate for inference
        """
        self.simulation_periods = simulation_periods

        # Initialize agents
        self.market_maker = MarketMaker(risk_aversion=market_maker_risk_aversion)
        self.arbitrageur = Arbitrageur(risk_aversion=arbitrageur_risk_aversion)
        self.noise_trader = NoiseTrader(risk_aversion=noise_trader_risk_aversion)

        self.agents = [self.market_maker, self.arbitrageur, self.noise_trader]

        logger.info("MultiAgentForecaster initialized with 3 agent types")

    def forecast(self, historical_prices: np.ndarray) -> Dict[str, float]:
        """
        Forecast market parameters using structural agent-based approach.

        Parameters:
        -----------
        historical_prices : np.ndarray
            Historical price time series

        Returns:
        --------
        forecast : Dict
            {
                'implied_volatility': float,  # From MarketMaker
                'implied_drift': float,       # From Arbitrageur
                'regime': str,                # From NoiseTrader
                'confidence': float           # Forecast confidence
            }
        """
        if len(historical_prices) < 20:
            logger.warning("Insufficient historical data for agent simulation")
            return self._default_forecast()

        # Simulate agent interactions
        simulation_results = simulate_agent_interaction(
            agents=self.agents,
            market_data=historical_prices,
            n_periods=min(self.simulation_periods, len(historical_prices))
        )

        # Extract inferred parameters
        inferred_params = simulation_results['parameters']

        # Aggregate into forecast
        implied_volatility = inferred_params.get('MarketMaker', 0.5)
        implied_drift = inferred_params.get('Arbitrageur', 0.0)
        regime = inferred_params.get('NoiseTrader', 'normal')

        # Compute confidence based on agent agreement
        confidence = self._compute_confidence(simulation_results)

        forecast = {
            'implied_volatility': implied_volatility,
            'implied_drift': implied_drift,
            'regime': regime,
            'confidence': confidence,
            'method': 'structural_multi_agent'
        }

        logger.info(f"Structural forecast: σ={implied_volatility:.3f}, μ={implied_drift:.3f}, regime={regime}")

        return forecast

    def _compute_confidence(self, simulation_results: Dict) -> float:
        """
        Compute forecast confidence based on agent behavior consistency.

        High confidence when:
        - Low variance in market maker spreads
        - Consistent arbitrageur direction
        - Stable noise trader sentiment
        """
        # Market maker spread stability
        mm_actions = simulation_results['actions'].get('MarketMaker', [])
        if len(mm_actions) > 1:
            spreads = [a.get('spread', 0.01) for a in mm_actions]
            spread_cv = np.std(spreads) / (np.mean(spreads) + 1e-6)  # Coefficient of variation
            mm_confidence = np.exp(-spread_cv * 5)  # Lower CV → higher confidence
        else:
            mm_confidence = 0.5

        # Arbitrageur trade consistency
        arb_actions = simulation_results['actions'].get('Arbitrageur', [])
        if len(arb_actions) > 0:
            buys = sum(1 for a in arb_actions if a['type'] == 'buy')
            sells = sum(1 for a in arb_actions if a['type'] == 'sell')
            total = buys + sells
            if total > 0:
                directional_consistency = abs(buys - sells) / total
                arb_confidence = directional_consistency
            else:
                arb_confidence = 0.3
        else:
            arb_confidence = 0.5

        # Overall confidence (weighted average)
        confidence = 0.5 * mm_confidence + 0.5 * arb_confidence

        return np.clip(confidence, 0.0, 1.0)

    def _default_forecast(self) -> Dict[str, float]:
        """Return default forecast when insufficient data"""
        return {
            'implied_volatility': 0.5,
            'implied_drift': 0.0,
            'regime': 'normal',
            'confidence': 0.3,
            'method': 'structural_multi_agent'
        }

    def explain_forecast(self, historical_prices: np.ndarray) -> str:
        """
        Generate human-readable explanation of forecast.

        This is a key advantage of structural models: interpretability.
        """
        forecast = self.forecast(historical_prices)

        explanation = []
        explanation.append("=" * 70)
        explanation.append(" STRUCTURAL FORECAST EXPLANATION (Multi-Agent Model)")
        explanation.append("=" * 70)

        explanation.append(f"\n📊 Forecast Results:")
        explanation.append(f"  • Implied Volatility: {forecast['implied_volatility']:.2%}")
        explanation.append(f"  • Implied Drift:      {forecast['implied_drift']:+.2%}")
        explanation.append(f"  • Regime:             {forecast['regime']}")
        explanation.append(f"  • Confidence:         {forecast['confidence']:.1%}")

        explanation.append(f"\n🤖 Agent Insights:")

        # Market Maker
        mm_vol = forecast['implied_volatility']
        if mm_vol > 0.7:
            mm_insight = "Market makers are widening spreads significantly → High uncertainty"
        elif mm_vol < 0.3:
            mm_insight = "Market makers are maintaining tight spreads → Low risk perception"
        else:
            mm_insight = "Market makers are quoting moderate spreads → Normal conditions"
        explanation.append(f"  • Market Maker: {mm_insight}")

        # Arbitrageur
        arb_drift = forecast['implied_drift']
        if arb_drift > 0.1:
            arb_insight = "Arbitrageurs are actively buying → Positive momentum detected"
        elif arb_drift < -0.1:
            arb_insight = "Arbitrageurs are actively selling → Negative momentum detected"
        else:
            arb_insight = "Arbitrageurs are mostly inactive → No strong trend"
        explanation.append(f"  • Arbitrageur: {arb_insight}")

        # Noise Trader
        regime = forecast['regime']
        if regime == 'high_vol':
            nt_insight = "Noise traders are clustering → Stress/high volatility regime"
        else:
            nt_insight = "Noise traders are dispersed → Normal market regime"
        explanation.append(f"  • Noise Trader: {nt_insight}")

        explanation.append(f"\n💡 Structural Interpretation:")
        explanation.append(f"   These parameters emerge from simulated agent behaviors,")
        explanation.append(f"   not direct statistical fitting. This provides:")
        explanation.append(f"   - Causal interpretation (why, not just what)")
        explanation.append(f"   - Robustness to spurious correlations")
        explanation.append(f"   - Adaptability to regime changes")

        explanation.append("\n" + "=" * 70)

        return "\n".join(explanation)

    def compare_with_reduced_form(self,
                                   historical_prices: np.ndarray,
                                   reduced_form_forecast: Dict) -> Dict:
        """
        Compare structural forecast with reduced-form (LSTM/GARCH).

        Parameters:
        -----------
        historical_prices : np.ndarray
        reduced_form_forecast : Dict
            {'volatility': float, 'drift': float}

        Returns:
        --------
        comparison : Dict
            Detailed comparison metrics
        """
        structural_forecast = self.forecast(historical_prices)

        vol_diff = abs(structural_forecast['implied_volatility'] -
                       reduced_form_forecast.get('volatility', 0.5))
        drift_diff = abs(structural_forecast['implied_drift'] -
                         reduced_form_forecast.get('drift', 0.0))

        comparison = {
            'structural': structural_forecast,
            'reduced_form': reduced_form_forecast,
            'volatility_difference': vol_diff,
            'drift_difference': drift_diff,
            'structural_confidence': structural_forecast['confidence'],
            'regime_detected': structural_forecast['regime']
        }

        return comparison
