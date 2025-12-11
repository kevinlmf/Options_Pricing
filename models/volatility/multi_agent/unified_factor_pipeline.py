"""
Unified Factor Pipeline
========================

Integrates all factor extraction and analysis approaches:
1. Observable Factor Extraction (from real market data)
2. Latent Factor Extraction (from simulated agents)
3. Factor Importance Analysis (identify important factors)
4. Agent Behavior Analysis (trace factors to agent behaviors)
5. Predictive Modeling (build models based on insights)

This unified pipeline provides:
- Flexibility: Use observable or latent factors
- Intelligence: Automatically identify important factors
- Integration: Seamlessly combine different approaches
- Better Predictions: Use insights to build better models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .observable_factor_extractor import ObservableFactorExtractor
from .agent_volatility import MultiAgentVolatilityExtractor
from .factor_importance_analyzer import FactorImportanceAnalyzer, FactorAnalysisResult
from .factor_modeling import MultiAgentFactorModeler


@dataclass
class UnifiedFactorResult:
    """Results from unified factor pipeline"""
    factors: pd.DataFrame  # Extracted factors
    factor_importance: Optional[FactorAnalysisResult] = None  # Importance analysis
    agent_behaviors: Optional[pd.DataFrame] = None  # Agent behavior metrics
    predictive_model: Optional[object] = None  # Trained model
    model_metrics: Optional[Dict[str, float]] = None  # Model performance
    top_factors: List[str] = None  # Most important factors
    factor_type: str = "observable"  # "observable" or "latent"


class UnifiedFactorPipeline:
    """
    Unified pipeline that integrates:
    - Observable factor extraction
    - Latent factor extraction  
    - Factor importance analysis
    - Predictive modeling
    
    This provides a seamless workflow from data to predictions.
    """
    
    def __init__(self,
                 use_observable_factors: bool = True,
                 analyze_importance: bool = True,
                 n_top_factors: int = 3):
        """
        Initialize unified pipeline.
        
        Parameters:
        -----------
        use_observable_factors : bool
            If True, use observable factors from real market data
            If False, use latent factors from simulated agents
        analyze_importance : bool
            If True, perform factor importance analysis
        n_top_factors : int
            Number of top factors to identify
        """
        self.use_observable_factors = use_observable_factors
        self.analyze_importance = analyze_importance
        self.n_top_factors = n_top_factors
        
        # Initialize components
        if use_observable_factors:
            self.observable_extractor = ObservableFactorExtractor()
        else:
            self.latent_extractor = MultiAgentVolatilityExtractor()
            self.latent_modeler = MultiAgentFactorModeler()
        
        if analyze_importance:
            self.importance_analyzer = FactorImportanceAnalyzer(n_top_factors=n_top_factors)
    
    def extract_factors(self,
                       market_data: pd.DataFrame,
                       agent_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Extract factors using the selected method.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with OHLCV columns
        agent_config : Dict, optional
            Configuration for latent factor extraction
            
        Returns:
        --------
        pd.DataFrame : Extracted factors
        """
        if self.use_observable_factors:
            # Extract observable factors from real market data
            factors = self.observable_extractor.extract_all_factors(market_data)
            # Remove combined_volatility if exists (we'll use individual factors)
            if 'combined_volatility' in factors.columns:
                factors = factors.drop(columns=['combined_volatility'])
            return factors
        else:
            # Extract latent factors from simulated agents
            if agent_config is None:
                # Use default configuration
                from .factor_modeling import AgentConfiguration
                agent_config = AgentConfiguration(
                    n_market_makers=25,
                    n_arbitrageurs=25,
                    n_trend_followers=25,
                    n_fundamental_investors=25,
                    weight_market_maker=0.25,
                    weight_arbitrageur=0.25,
                    weight_trend_follower=0.25,
                    weight_fundamental=0.25
                )
            
            # Run simulation and extract factors
            results = self.latent_modeler._run_simulation_with_config(
                agent_config,
                n_steps=len(market_data) - 1
            )
            
            # Extract factor columns
            factor_cols = [col for col in results.columns if col.endswith('_vol')]
            return results[factor_cols]
    
    def extract_agent_behaviors(self,
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract agent behavior metrics from market data.
        
        For observable factors, these are derived from market microstructure.
        For latent factors, these come from simulated agent states.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data
            
        Returns:
        --------
        pd.DataFrame : Agent behavior metrics
        """
        behaviors = pd.DataFrame(index=market_data.index)
        
        # Market Maker behaviors
        behaviors['spread'] = (market_data['High'] - market_data['Low']) / market_data['Close']
        behaviors['bid_ask_spread'] = behaviors['spread']
        
        # Arbitrageur behaviors
        behaviors['trading_frequency'] = market_data['Volume'] / market_data['Volume'].rolling(20).mean()
        behaviors['trade_count'] = market_data['Volume']
        behaviors['volume'] = market_data['Volume']
        
        # Trend Follower behaviors
        returns = market_data['Close'].pct_change()
        behaviors['momentum'] = returns.rolling(5).mean()
        behaviors['trend_strength'] = abs(returns.rolling(20).mean())
        behaviors['herding_ratio'] = (returns > 0).rolling(20).mean()
        
        # Fundamental behaviors
        behaviors['volume_price_ratio'] = market_data['Volume'] / market_data['Close']
        behaviors['dollar_volume'] = market_data['Volume'] * market_data['Close']
        behaviors['holdings_ratio'] = (market_data['Volume'] > market_data['Volume'].rolling(20).mean()).astype(float)
        
        return behaviors.fillna(method='ffill').fillna(0)
    
    def calculate_target_volatility(self,
                                   market_data: pd.DataFrame,
                                   window: int = 20) -> pd.Series:
        """
        Calculate realized volatility as target variable.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with 'Close' column
        window : int
            Rolling window for volatility calculation
            
        Returns:
        --------
        pd.Series : Realized volatility
        """
        returns = market_data['Close'].pct_change()
        realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return realized_vol.dropna()
    
    def run_complete_pipeline(self,
                            market_data: pd.DataFrame,
                            agent_config: Optional[Dict] = None) -> UnifiedFactorResult:
        """
        Run complete unified pipeline:
        1. Extract factors
        2. Extract agent behaviors
        3. Calculate target volatility
        4. Analyze factor importance (if enabled)
        5. Build predictive model (if importance analysis enabled)
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            Market data with OHLCV columns
        agent_config : Dict, optional
            Configuration for latent factor extraction
            
        Returns:
        --------
        UnifiedFactorResult : Complete pipeline results
        """
        # Step 1: Extract factors
        factors = self.extract_factors(market_data, agent_config)
        
        # Step 2: Extract agent behaviors
        agent_behaviors = self.extract_agent_behaviors(market_data)
        
        # Step 3: Calculate target volatility
        target_volatility = self.calculate_target_volatility(market_data)
        
        # Initialize result
        result = UnifiedFactorResult(
            factors=factors,
            agent_behaviors=agent_behaviors,
            factor_type="observable" if self.use_observable_factors else "latent"
        )
        
        # Step 4: Factor importance analysis (if enabled)
        if self.analyze_importance:
            # Align all data
            common_idx = factors.index.intersection(
                agent_behaviors.index
            ).intersection(target_volatility.index)
            
            if len(common_idx) > 20:  # Need enough data
                factors_aligned = factors.loc[common_idx]
                behaviors_aligned = agent_behaviors.loc[common_idx]
                target_aligned = target_volatility.loc[common_idx]
                
                # Run importance analysis
                importance_result = self.importance_analyzer.complete_analysis(
                    factors_aligned,
                    behaviors_aligned,
                    target_aligned
                )
                
                result.factor_importance = importance_result
                result.top_factors = importance_result.top_factors
                
                # Step 5: Build predictive model
                model, metrics = self.importance_analyzer.build_predictive_model(
                    factors_aligned,
                    target_aligned,
                    use_top_factors_only=True
                )
                
                result.predictive_model = model
                result.model_metrics = metrics
            else:
                print(f"Warning: Not enough overlapping data ({len(common_idx)} points). Skipping importance analysis.")
        
        return result
    
    def predict_volatility(self,
                         result: UnifiedFactorResult,
                         market_data: pd.DataFrame) -> pd.Series:
        """
        Predict volatility using the trained model.
        
        Parameters:
        -----------
        result : UnifiedFactorResult
            Results from run_complete_pipeline
        market_data : pd.DataFrame
            Market data for prediction
            
        Returns:
        --------
        pd.Series : Predicted volatility
        """
        if result.predictive_model is None:
            raise ValueError("No predictive model available. Run pipeline with analyze_importance=True")
        
        # Extract factors for prediction
        factors = self.extract_factors(market_data)
        
        # Use top factors if available
        if result.top_factors:
            factors = factors[result.top_factors]
        
        # Predict
        predictions = result.predictive_model.predict(factors.values)
        
        return pd.Series(predictions, index=factors.index)


if __name__ == "__main__":
    # Example usage
    from data.market_data import YahooDataProvider
    
    # Load data
    provider = YahooDataProvider()
    data = provider.get_price_data('AAPL', '2023-01-01', '2024-01-01')
    
    if not data.empty:
        # Run unified pipeline with observable factors
        pipeline = UnifiedFactorPipeline(
            use_observable_factors=True,
            analyze_importance=True,
            n_top_factors=3
        )
        
        result = pipeline.run_complete_pipeline(data)
        
        print("Unified Pipeline Results:")
        print(f"Factor Type: {result.factor_type}")
        print(f"Factors Extracted: {list(result.factors.columns)}")
        
        if result.factor_importance:
            print(f"\nTop Factors: {result.top_factors}")
            print(f"\nModel Metrics: {result.model_metrics}")


