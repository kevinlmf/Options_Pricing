"""
Agent-Physical Model Integration
=================================

Integrates multi-agent factor extraction with physical models (jump-diffusion)
using time-series modeling to ensure convergence.

Complete Pipeline:
1. Extract factors from multi-agent interactions (micro-mechanism)
2. Fit physical model (jump-diffusion) to historical data (macro-constraint)
3. Use time-series modeling to connect micro and macro
4. Ensure factors converge to physical model predictions
5. Predict volatility using constrained factors

Time-Series Role:
- Models factor dynamics: dF_t = κ(θ - F_t)dt + σ_F dW_t
- Connects agent behaviors to physical model parameters
- Ensures long-term convergence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .unified_factor_pipeline import UnifiedFactorPipeline, UnifiedFactorResult
from .physical_model_constraint import PhysicalModelConstraint, JumpDiffusionModel


@dataclass
class IntegratedVolatilityPrediction:
    """Integrated prediction results"""
    agent_factors: pd.DataFrame  # Factors from multi-agent
    physical_model_target: float  # Long-term volatility from physical model
    constrained_factors: pd.DataFrame  # Factors constrained to converge
    predicted_volatility: pd.Series  # Final prediction
    convergence_metrics: Dict[str, float]  # How well factors converge
    factor_physical_mapping: Dict[str, float]  # How factors map to physical model


class AgentPhysicalIntegration:
    """
    Integrates multi-agent factors with physical models.
    
    Key Components:
    1. Multi-Agent: Provides micro-mechanism (agent behavior → factors)
    2. Physical Model: Provides macro-constraint (jump-diffusion → long-term vol)
    3. Time-Series: Connects micro and macro (factor dynamics → convergence)
    """
    
    def __init__(self,
                 use_observable_factors: bool = True,
                 physical_model: str = 'jump_diffusion',
                 convergence_rate: float = 0.1):
        """
        Initialize integration.
        
        Parameters:
        -----------
        use_observable_factors : bool
            Use observable factors from real data
        physical_model : str
            Type of physical model ('jump_diffusion', etc.)
        convergence_rate : float
            Rate of convergence to physical model
        """
        self.use_observable_factors = use_observable_factors
        
        # Multi-agent factor pipeline
        self.factor_pipeline = UnifiedFactorPipeline(
            use_observable_factors=use_observable_factors,
            analyze_importance=True,
            n_top_factors=3
        )
        
        # Physical model constraint
        self.physical_constraint = PhysicalModelConstraint(
            physical_model=physical_model,
            convergence_rate=convergence_rate
        )
    
    def run_integrated_pipeline(self,
                               market_data: pd.DataFrame) -> IntegratedVolatilityPrediction:
        """
        Run complete integrated pipeline:
        1. Extract factors from multi-agent interactions
        2. Fit physical model to historical data
        3. Map factors to physical model parameters (time-series)
        4. Apply convergence constraint
        5. Predict volatility
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with OHLCV columns
            
        Returns:
        --------
        IntegratedVolatilityPrediction : Complete results
        """
        # Step 1: Extract factors from multi-agent
        print("Step 1: Extracting factors from multi-agent interactions...")
        factor_result = self.factor_pipeline.run_complete_pipeline(market_data)
        agent_factors = factor_result.factors
        
        print(f"  ✓ Extracted {len(agent_factors.columns)} factors")
        if factor_result.top_factors:
            print(f"  ✓ Top factors: {factor_result.top_factors}")
        
        # Step 2: Fit physical model (jump-diffusion)
        print("\nStep 2: Fitting physical model (jump-diffusion)...")
        returns = market_data['Close'].pct_change().dropna()
        physical_fit = self.physical_constraint.fit_physical_model(returns.values)
        
        physical_target = physical_fit.parameters['long_term_volatility']
        print(f"  ✓ Long-term volatility target: {physical_target:.4f}")
        print(f"  ✓ Jump intensity: {physical_fit.parameters['lambda_jump']:.4f}")
        print(f"  ✓ Diffusion volatility: {physical_fit.parameters['sigma']:.4f}")
        
        # Step 3: Map factors to physical model (time-series regression)
        print("\nStep 3: Mapping factors to physical model (time-series)...")
        target_volatility = returns.rolling(20).std() * np.sqrt(252)
        target_volatility = target_volatility.dropna()  # Remove NaN values from rolling calculation

        factor_mapping = self.physical_constraint.map_factors_to_physical_model(
            agent_factors,
            target_volatility
        )
        
        print(f"  ✓ Factor-Physical Model Mapping:")
        for factor, loading in factor_mapping.items():
            print(f"     {factor}: {loading:.4f}")
        
        # Step 4: Apply convergence constraint (focus on long-term physical model convergence)
        print("\nStep 4: Applying convergence constraint...")
        print("  - Long-term: Converge to physical model target")
        print("  - Note: Dual convergence (short-term + long-term) will be added in next phase")

        # For now, use simple convergence to physical model target
        constrained_factors = self.physical_constraint.constrained_factor_dynamics(
            agent_factors,
            physical_target
        )
        
        # Step 5: Check convergence
        convergence_metrics = self.physical_constraint.check_convergence(
            constrained_factors
        )
        
        print(f"  ✓ Convergence metrics:")
        for factor, metric in convergence_metrics.items():
            print(f"     {factor}: {metric:.4f}")
        
        # Step 6: Predict volatility using constrained factors
        print("\nStep 5: Predicting volatility...")
        
        # Use constrained factors for prediction
        if factor_result.predictive_model:
            # Use top factors if available
            if factor_result.top_factors:
                prediction_factors = constrained_factors[factor_result.top_factors]
            else:
                prediction_factors = constrained_factors
            
            predicted_vol = pd.Series(
                factor_result.predictive_model.predict(prediction_factors.values),
                index=prediction_factors.index
            )
        else:
            # Fallback: weighted average of constrained factors
            if factor_result.top_factors:
                weights = np.ones(len(factor_result.top_factors)) / len(factor_result.top_factors)
                predicted_vol = (
                    constrained_factors[factor_result.top_factors] * weights
                ).sum(axis=1)
            else:
                predicted_vol = constrained_factors.mean(axis=1)
        
        print(f"  ✓ Predicted volatility range: [{predicted_vol.min():.4f}, {predicted_vol.max():.4f}]")
        
        return IntegratedVolatilityPrediction(
            agent_factors=agent_factors,
            physical_model_target=physical_target,
            constrained_factors=constrained_factors,
            predicted_volatility=predicted_vol,
            convergence_metrics=convergence_metrics,
            factor_physical_mapping=factor_mapping
        )
    
    def explain_agent_behavior_impact(self,
                                     result: IntegratedVolatilityPrediction,
                                     factor_result: UnifiedFactorResult) -> Dict[str, Dict]:
        """
        Explain how agent behaviors impact volatility through factors.
        
        Returns:
        --------
        Dict : Explanation of agent behavior → factor → volatility chain
        """
        explanations = {}
        
        if factor_result.factor_importance and factor_result.factor_importance.top_agent_behaviors:
            for factor_name in result.agent_factors.columns:
                if factor_name in factor_result.factor_importance.top_agent_behaviors:
                    behaviors = factor_result.factor_importance.top_agent_behaviors[factor_name]
                    
                    # Get physical model contribution
                    physical_loading = result.factor_physical_mapping.get(factor_name, 0.0)
                    
                    explanations[factor_name] = {
                        'agent_behaviors': behaviors,
                        'physical_model_loading': physical_loading,
                        'convergence_metric': result.convergence_metrics.get(factor_name, 0.0),
                        'impact': 'High' if abs(physical_loading) > 0.1 else 'Low'
                    }
        
        return explanations


if __name__ == "__main__":
    # Example usage
    from data.market_data import YahooDataProvider
    
    provider = YahooDataProvider()
    data = provider.get_price_data('AAPL', '2023-01-01', '2024-01-01')
    
    if not data.empty:
        integrator = AgentPhysicalIntegration(
            use_observable_factors=True,
            physical_model='jump_diffusion',
            convergence_rate=0.1
        )
        
        result = integrator.run_integrated_pipeline(data)
        
        print("\n" + "=" * 80)
        print("Integration Results")
        print("=" * 80)
        print(f"Physical Model Target: {result.physical_model_target:.4f}")
        print(f"Predicted Volatility (mean): {result.predicted_volatility.mean():.4f}")
        print(f"Convergence: {np.mean(list(result.convergence_metrics.values())):.4f}")

