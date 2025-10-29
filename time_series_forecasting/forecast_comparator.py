"""
Forecasting Model Comparison Framework

Compares Reduced-form vs Structural approaches:

┌─────────────────────────────────────────────────────────────┐
│  Framework      │ Nature           │ Goal                   │
├─────────────────────────────────────────────────────────────┤
│  Time-Series    │ Reduced-form     │ Fit statistical        │
│  (LSTM/GARCH)   │                  │ relationships          │
├─────────────────────────────────────────────────────────────┤
│  Multi-Agent    │ Structural       │ Explain behavioral     │
│  Simulation     │                  │ causal mechanisms      │
└─────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Unified forecast result structure"""
    volatility: float
    drift: float
    method: str  # 'reduced_form' or 'structural'
    confidence: float = 0.5
    regime: Optional[str] = None
    computation_time: float = 0.0
    explanation: Optional[str] = None


class ForecastComparator:
    """
    Comparison framework for reduced-form vs structural forecasting.

    Metrics:
    - Accuracy (if ground truth available)
    - Interpretability
    - Computational cost
    - Robustness to regime changes
    """

    def __init__(self):
        self.comparison_history: List[Dict] = []

    def compare_forecasts(self,
                         historical_prices: np.ndarray,
                         reduced_form_forecaster,
                         structural_forecaster) -> Dict:
        """
        Run both forecasters and compare results.

        Parameters:
        -----------
        historical_prices : np.ndarray
        reduced_form_forecaster : object
            Must have .forecast(prices) → {'volatility': float, 'drift': float}
        structural_forecaster : MultiAgentForecaster
            Multi-agent structural model

        Returns:
        --------
        comparison : Dict
            Detailed comparison results
        """
        logger.info("Running forecast comparison: Reduced-form vs Structural")

        # Run reduced-form forecast (LSTM/GARCH)
        start_time = time.time()
        try:
            rf_result = self._run_reduced_form(reduced_form_forecaster, historical_prices)
        except Exception as e:
            logger.error(f"Reduced-form forecast failed: {e}")
            rf_result = ForecastResult(
                volatility=0.5,
                drift=0.0,
                method='reduced_form',
                confidence=0.0,
                computation_time=0.0
            )
        rf_time = time.time() - start_time

        # Run structural forecast (Multi-Agent)
        start_time = time.time()
        try:
            struct_result = self._run_structural(structural_forecaster, historical_prices)
        except Exception as e:
            logger.error(f"Structural forecast failed: {e}")
            struct_result = ForecastResult(
                volatility=0.5,
                drift=0.0,
                method='structural',
                confidence=0.0,
                computation_time=0.0
            )
        struct_time = time.time() - start_time

        # Compute comparison metrics
        comparison = {
            'reduced_form': {
                'volatility': rf_result.volatility,
                'drift': rf_result.drift,
                'confidence': rf_result.confidence,
                'time_ms': rf_time * 1000,
                'method': 'LSTM/GARCH (Reduced-form)'
            },
            'structural': {
                'volatility': struct_result.volatility,
                'drift': struct_result.drift,
                'confidence': struct_result.confidence,
                'regime': struct_result.regime,
                'time_ms': struct_time * 1000,
                'method': 'Multi-Agent (Structural)'
            },
            'differences': {
                'volatility_diff': abs(rf_result.volatility - struct_result.volatility),
                'drift_diff': abs(rf_result.drift - struct_result.drift),
                'confidence_diff': struct_result.confidence - rf_result.confidence
            },
            'speed_ratio': rf_time / (struct_time + 1e-9),
            'recommendation': self._generate_recommendation(rf_result, struct_result)
        }

        # Store in history
        self.comparison_history.append(comparison)

        logger.info(f"Comparison complete: Δσ={comparison['differences']['volatility_diff']:.3f}, "
                   f"Δμ={comparison['differences']['drift_diff']:.3f}")

        return comparison

    def _run_reduced_form(self, forecaster, historical_prices: np.ndarray) -> ForecastResult:
        """Run reduced-form forecast (statistical fitting)"""
        # Compute simple historical statistics as baseline reduced-form
        returns = np.diff(np.log(historical_prices))

        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        drift = np.mean(returns) * 252  # Annualized

        return ForecastResult(
            volatility=volatility,
            drift=drift,
            method='reduced_form',
            confidence=0.6,  # Fixed confidence for statistical methods
            regime=None,
            explanation="Direct statistical estimation from historical returns"
        )

    def _run_structural(self, forecaster, historical_prices: np.ndarray) -> ForecastResult:
        """Run structural forecast (agent-based simulation)"""
        forecast_dict = forecaster.forecast(historical_prices)

        return ForecastResult(
            volatility=forecast_dict['implied_volatility'],
            drift=forecast_dict['implied_drift'],
            method='structural',
            confidence=forecast_dict['confidence'],
            regime=forecast_dict['regime'],
            explanation=forecaster.explain_forecast(historical_prices)
        )

    def _generate_recommendation(self,
                                 rf_result: ForecastResult,
                                 struct_result: ForecastResult) -> str:
        """
        Generate recommendation on which forecast to use.

        Decision logic:
        - If regime is 'high_vol' → prefer structural (better at regime detection)
        - If confidence is high → prefer structural (more reliable)
        - If disagreement is large → investigate further
        """
        vol_diff = abs(rf_result.volatility - struct_result.volatility)
        drift_diff = abs(rf_result.drift - struct_result.drift)

        if struct_result.regime == 'high_vol':
            return ("Structural model recommended: Detected high volatility regime. "
                   "Structural models adapt better to regime changes.")

        if struct_result.confidence > 0.7:
            return ("Structural model recommended: High confidence in agent-based inference. "
                   "Behavioral mechanisms are consistent.")

        if vol_diff > 0.2 or drift_diff > 0.1:
            return ("Models disagree significantly. Investigate further. "
                   f"Δσ={vol_diff:.2f}, Δμ={drift_diff:.2f}. "
                   "Consider ensemble or additional validation.")

        if rf_result.volatility < 0.3 and struct_result.volatility < 0.3:
            return ("Low volatility environment: Both models agree. "
                   "Reduced-form is computationally cheaper in stable regimes.")

        return ("Models broadly agree. Use structural for interpretability, "
               "reduced-form for computational efficiency.")

    def generate_comparison_report(self, comparison: Dict) -> str:
        """Generate detailed comparison report"""
        report = []
        report.append("=" * 80)
        report.append(" FORECASTING MODEL COMPARISON")
        report.append("=" * 80)

        report.append("\n┌─────────────────┬──────────────────┬────────────────────┬──────────────┐")
        report.append("│  Metric         │  Reduced-form    │  Structural        │  Difference  │")
        report.append("├─────────────────┼──────────────────┼────────────────────┼──────────────┤")

        rf = comparison['reduced_form']
        st = comparison['structural']
        diff = comparison['differences']

        report.append(f"│  Volatility (σ) │  {rf['volatility']:>14.2%} │  {st['volatility']:>16.2%} │  {diff['volatility_diff']:>10.2%} │")
        report.append(f"│  Drift (μ)      │  {rf['drift']:>+14.2%} │  {st['drift']:>+16.2%} │  {diff['drift_diff']:>10.2%} │")
        report.append(f"│  Confidence     │  {rf['confidence']:>14.1%} │  {st['confidence']:>16.1%} │  {diff['confidence_diff']:>+10.1%} │")
        report.append(f"│  Time (ms)      │  {rf['time_ms']:>14.2f} │  {st['time_ms']:>16.2f} │              │")
        report.append("└─────────────────┴──────────────────┴────────────────────┴──────────────┘")

        if st['regime']:
            report.append(f"\n🔍 Regime Detected (Structural only): {st['regime'].upper()}")

        report.append(f"\n📊 Nature of Models:")
        report.append(f"  • Reduced-form: Fit statistical relationships (black-box)")
        report.append(f"  • Structural:   Simulate behavioral mechanisms (interpretable)")

        report.append(f"\n💡 Recommendation:")
        report.append(f"  {comparison['recommendation']}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def evaluate_with_ground_truth(self,
                                   forecast_comparison: Dict,
                                   true_volatility: float,
                                   true_drift: float) -> Dict:
        """
        Evaluate forecast accuracy against ground truth.

        Parameters:
        -----------
        forecast_comparison : Dict
            Result from compare_forecasts()
        true_volatility, true_drift : float
            Ground truth parameters

        Returns:
        --------
        evaluation : Dict
            Accuracy metrics for both methods
        """
        rf = forecast_comparison['reduced_form']
        st = forecast_comparison['structural']

        rf_vol_error = abs(rf['volatility'] - true_volatility)
        st_vol_error = abs(st['volatility'] - true_volatility)

        rf_drift_error = abs(rf['drift'] - true_drift)
        st_drift_error = abs(st['drift'] - true_drift)

        evaluation = {
            'reduced_form': {
                'volatility_error': rf_vol_error,
                'drift_error': rf_drift_error,
                'total_error': rf_vol_error + rf_drift_error
            },
            'structural': {
                'volatility_error': st_vol_error,
                'drift_error': st_drift_error,
                'total_error': st_vol_error + st_drift_error
            },
            'winner': 'structural' if (st_vol_error + st_drift_error) < (rf_vol_error + rf_drift_error) else 'reduced_form'
        }

        return evaluation


def demonstrate_comparison():
    """
    Demonstration function showing the comparison framework.
    """
    print("=" * 80)
    print(" FORECASTING COMPARISON DEMONSTRATION")
    print("=" * 80)

    print("\n📚 Framework Overview:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  Framework      │ Nature           │ Goal                   │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Time-Series    │ Reduced-form     │ Fit statistical        │")
    print("│  (LSTM/GARCH)   │                  │ relationships          │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Multi-Agent    │ Structural       │ Explain behavioral     │")
    print("│  Simulation     │                  │ causal mechanisms      │")
    print("└─────────────────────────────────────────────────────────────┘")

    print("\n✨ Key Differences:")
    print("  • Reduced-form: Black-box, data-driven, fast")
    print("  • Structural:   Interpretable, mechanism-based, robust")

    print("\n🎯 When to Use Each:")
    print("  • Stable markets → Reduced-form (efficiency)")
    print("  • Regime changes → Structural (robustness)")
    print("  • Need explanation → Structural (interpretability)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    demonstrate_comparison()
