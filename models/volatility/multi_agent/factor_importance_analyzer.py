"""
Factor Importance Analyzer
==========================

Systematic approach to volatility prediction:
1. Identify the most important factors affecting volatility
2. Trace back to agent behaviors that generate these factors
3. Analyze covariance/variance relationships between factors and agents
4. Build predictive models based on these insights

Key Insight:
- Not all factors are equally important
- Understanding agent behavior → factor causality improves prediction
- Covariance analysis reveals factor interactions and agent contributions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@dataclass
class FactorImportance:
    """Factor importance analysis results"""
    factor_name: str
    importance_score: float  # Feature importance from model
    correlation_with_volatility: float
    variance_explained: float  # R² contribution
    agent_behaviors: Dict[str, float]  # Which agent behaviors contribute to this factor


@dataclass
class AgentFactorRelationship:
    """Relationship between agent behavior and factor"""
    agent_type: str
    behavior_metric: str  # e.g., 'spread', 'trading_frequency', 'herding_ratio'
    factor_name: str
    covariance: float
    correlation: float
    variance_contribution: float  # How much variance in factor is explained by agent


@dataclass
class FactorAnalysisResult:
    """Complete factor analysis results"""
    factor_importances: List[FactorImportance]
    agent_factor_relationships: List[AgentFactorRelationship]
    factor_covariance_matrix: pd.DataFrame
    agent_covariance_matrix: pd.DataFrame
    factor_agent_covariance: pd.DataFrame
    top_factors: List[str]  # Most important factors
    top_agent_behaviors: Dict[str, List[str]]  # Top behaviors for each factor


class FactorImportanceAnalyzer:
    """
    Analyze factor importance and agent behavior relationships.
    
    Pipeline:
    1. Extract factors and agent behaviors
    2. Identify most important factors for volatility prediction
    3. Trace factors back to agent behaviors
    4. Analyze covariance/variance relationships
    5. Build predictive models
    """
    
    def __init__(self, 
                 n_top_factors: int = 3,
                 use_random_forest: bool = True):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        n_top_factors : int
            Number of top factors to identify
        use_random_forest : bool
            Use Random Forest for importance analysis (more robust than correlation)
        """
        self.n_top_factors = n_top_factors
        self.use_random_forest = use_random_forest
        self.importance_model = None
        
    def analyze_factor_importance(self,
                                 factors: pd.DataFrame,
                                 target_volatility: pd.Series) -> List[FactorImportance]:
        """
        Identify the most important factors for volatility prediction.
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factor values over time (columns: market_maker_vol, arbitrageur_vol, etc.)
        target_volatility : pd.Series
            Realized volatility (target variable)
            
        Returns:
        --------
        List[FactorImportance] : Sorted by importance score
        """
        # Align indices
        common_idx = factors.index.intersection(target_volatility.index)
        factors_aligned = factors.loc[common_idx]
        target_aligned = target_volatility.loc[common_idx]
        
        if len(common_idx) < 10:
            raise ValueError("Not enough overlapping data points")
        
        factor_importances = []
        
        # Method 1: Random Forest Feature Importance (more robust)
        if self.use_random_forest:
            self.importance_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.importance_model.fit(factors_aligned.values, target_aligned.values)
            
            # Get feature importances
            importances = self.importance_model.feature_importances_
            
            # Calculate R² contribution for each factor
            r2_scores = []
            for i, factor_name in enumerate(factors_aligned.columns):
                # Train model with only this factor
                single_factor_model = RandomForestRegressor(n_estimators=50, random_state=42)
                single_factor_model.fit(
                    factors_aligned[[factor_name]].values,
                    target_aligned.values
                )
                r2 = r2_score(
                    target_aligned.values,
                    single_factor_model.predict(factors_aligned[[factor_name]].values)
                )
                r2_scores.append(r2)
            
            # Calculate correlations
            correlations = factors_aligned.corrwith(target_aligned)
            
            for i, factor_name in enumerate(factors_aligned.columns):
                factor_importances.append(FactorImportance(
                    factor_name=factor_name,
                    importance_score=importances[i],
                    correlation_with_volatility=correlations[factor_name],
                    variance_explained=r2_scores[i],
                    agent_behaviors={}  # Will be filled later
                ))
        
        # Method 2: Correlation-based (fallback)
        else:
            correlations = factors_aligned.corrwith(target_aligned).abs().sort_values(ascending=False)
            
            for factor_name, corr in correlations.items():
                # Simple R² approximation
                r2_approx = corr ** 2
                
                factor_importances.append(FactorImportance(
                    factor_name=factor_name,
                    importance_score=corr,
                    correlation_with_volatility=corr,
                    variance_explained=r2_approx,
                    agent_behaviors={}
                ))
        
        # Sort by importance score
        factor_importances.sort(key=lambda x: x.importance_score, reverse=True)
        
        return factor_importances
    
    def trace_factors_to_agent_behaviors(self,
                                        factors: pd.DataFrame,
                                        agent_behaviors: pd.DataFrame) -> List[AgentFactorRelationship]:
        """
        Trace factors back to agent behaviors that generate them.
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factor values (columns: market_maker_vol, arbitrageur_vol, etc.)
        agent_behaviors : pd.DataFrame
            Agent behavior metrics (columns: spread, trading_frequency, herding_ratio, etc.)
            
        Returns:
        --------
        List[AgentFactorRelationship] : Relationships between agents and factors
        """
        # Align indices
        common_idx = factors.index.intersection(agent_behaviors.index)
        factors_aligned = factors.loc[common_idx]
        behaviors_aligned = agent_behaviors.loc[common_idx]
        
        relationships = []
        
        # Map factor names to expected agent behaviors
        factor_agent_mapping = {
            'market_maker_vol': ['spread', 'bid_ask_spread'],
            'arbitrageur_vol': ['trading_frequency', 'trade_count', 'volume'],
            'trend_follower_vol': ['herding_ratio', 'momentum', 'trend_strength'],
            'fundamental_vol': ['holdings_ratio', 'volume_price_ratio', 'dollar_volume']
        }
        
        for factor_name in factors_aligned.columns:
            factor_values = factors_aligned[factor_name]
            
            # Find which agent behaviors correlate with this factor
            for behavior_col in behaviors_aligned.columns:
                if len(behaviors_aligned[behavior_col].dropna()) < 5:
                    continue
                
                behavior_values = behaviors_aligned[behavior_col]
                
                # Calculate covariance and correlation
                common_mask = factor_values.notna() & behavior_values.notna()
                if common_mask.sum() < 5:
                    continue
                
                factor_clean = factor_values[common_mask]
                behavior_clean = behavior_values[common_mask]
                
                # Covariance
                cov = np.cov(factor_clean, behavior_clean)[0, 1]
                
                # Correlation
                corr, p_value = stats.pearsonr(factor_clean, behavior_clean)
                
                # Variance contribution (R² from simple regression)
                if len(factor_clean) > 2:
                    # Simple linear regression
                    slope, intercept, r_value, p_val, std_err = stats.linregress(
                        behavior_clean, factor_clean
                    )
                    variance_contribution = r_value ** 2
                else:
                    variance_contribution = 0.0
                
                # Only include significant relationships
                if abs(corr) > 0.1 and p_value < 0.1:  # Thresholds
                    # Determine agent type from behavior name
                    agent_type = self._infer_agent_type(behavior_col)
                    
                    relationships.append(AgentFactorRelationship(
                        agent_type=agent_type,
                        behavior_metric=behavior_col,
                        factor_name=factor_name,
                        covariance=cov,
                        correlation=corr,
                        variance_contribution=variance_contribution
                    ))
        
        # Sort by absolute correlation
        relationships.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        return relationships
    
    def _infer_agent_type(self, behavior_name: str) -> str:
        """Infer agent type from behavior metric name"""
        behavior_lower = behavior_name.lower()
        
        if 'spread' in behavior_lower or 'bid' in behavior_lower or 'ask' in behavior_lower:
            return 'market_maker'
        elif 'trade' in behavior_lower or 'frequency' in behavior_lower or 'volume' in behavior_lower:
            return 'arbitrageur'
        elif 'herd' in behavior_lower or 'momentum' in behavior_lower or 'trend' in behavior_lower:
            return 'trend_follower'
        elif 'hold' in behavior_lower or 'fundamental' in behavior_lower or 'dollar' in behavior_lower:
            return 'fundamental'
        else:
            return 'unknown'
    
    def analyze_covariance_structure(self,
                                    factors: pd.DataFrame,
                                    agent_behaviors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Analyze covariance structure:
        1. Factor-Factor covariance
        2. Agent-Agent covariance
        3. Factor-Agent covariance
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factor values
        agent_behaviors : pd.DataFrame
            Agent behavior metrics
            
        Returns:
        --------
        Tuple of:
        - factor_covariance_matrix: pd.DataFrame
        - agent_covariance_matrix: pd.DataFrame
        - factor_agent_covariance: pd.DataFrame
        """
        # Align indices
        common_idx = factors.index.intersection(agent_behaviors.index)
        factors_aligned = factors.loc[common_idx]
        behaviors_aligned = agent_behaviors.loc[common_idx]
        
        # Factor-Factor covariance
        factor_cov = factors_aligned.cov()
        
        # Agent-Agent covariance
        agent_cov = behaviors_aligned.cov()
        
        # Factor-Agent covariance (cross-covariance)
        factor_agent_cov = pd.DataFrame(
            index=factors_aligned.columns,
            columns=behaviors_aligned.columns
        )
        
        for factor_col in factors_aligned.columns:
            for behavior_col in behaviors_aligned.columns:
                factor_vals = factors_aligned[factor_col]
                behavior_vals = behaviors_aligned[behavior_col]
                
                common_mask = factor_vals.notna() & behavior_vals.notna()
                if common_mask.sum() > 2:
                    cov_val = np.cov(
                        factor_vals[common_mask],
                        behavior_vals[common_mask]
                    )[0, 1]
                    factor_agent_cov.loc[factor_col, behavior_col] = cov_val
        
        return factor_cov, agent_cov, factor_agent_cov
    
    def complete_analysis(self,
                         factors: pd.DataFrame,
                         agent_behaviors: pd.DataFrame,
                         target_volatility: pd.Series) -> FactorAnalysisResult:
        """
        Complete factor importance analysis pipeline.
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factor values over time
        agent_behaviors : pd.DataFrame
            Agent behavior metrics over time
        target_volatility : pd.Series
            Realized volatility (target)
            
        Returns:
        --------
        FactorAnalysisResult : Complete analysis results
        """
        # Step 1: Identify most important factors
        factor_importances = self.analyze_factor_importance(factors, target_volatility)
        
        # Step 2: Trace factors to agent behaviors
        agent_relationships = self.trace_factors_to_agent_behaviors(factors, agent_behaviors)
        
        # Step 3: Analyze covariance structure
        factor_cov, agent_cov, factor_agent_cov = self.analyze_covariance_structure(
            factors, agent_behaviors
        )
        
        # Step 4: Identify top factors
        top_factors = [f.factor_name for f in factor_importances[:self.n_top_factors]]
        
        # Step 5: Identify top agent behaviors for each factor
        top_agent_behaviors = {}
        for factor_name in top_factors:
            factor_relationships = [
                r for r in agent_relationships 
                if r.factor_name == factor_name
            ]
            top_behaviors = sorted(
                factor_relationships,
                key=lambda x: abs(x.correlation),
                reverse=True
            )[:3]  # Top 3 behaviors per factor
            top_agent_behaviors[factor_name] = [
                f"{r.agent_type}.{r.behavior_metric}" 
                for r in top_behaviors
            ]
        
        # Update factor importances with agent behaviors
        for factor_imp in factor_importances:
            factor_relationships = [
                r for r in agent_relationships 
                if r.factor_name == factor_imp.factor_name
            ]
            factor_imp.agent_behaviors = {
                f"{r.agent_type}.{r.behavior_metric}": r.correlation
                for r in factor_relationships[:5]  # Top 5 behaviors
            }
        
        return FactorAnalysisResult(
            factor_importances=factor_importances,
            agent_factor_relationships=agent_relationships,
            factor_covariance_matrix=factor_cov,
            agent_covariance_matrix=agent_cov,
            factor_agent_covariance=factor_agent_cov,
            top_factors=top_factors,
            top_agent_behaviors=top_agent_behaviors
        )
    
    def build_predictive_model(self,
                              factors: pd.DataFrame,
                              target_volatility: pd.Series,
                              use_top_factors_only: bool = True) -> Tuple[object, Dict[str, float]]:
        """
        Build predictive model based on factor importance analysis.
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factor values
        target_volatility : pd.Series
            Target volatility
        use_top_factors_only : bool
            Use only top factors for prediction
            
        Returns:
        --------
        Tuple of:
        - model: Trained predictive model
        - metrics: Dictionary of performance metrics
        """
        # Align data
        common_idx = factors.index.intersection(target_volatility.index)
        factors_aligned = factors.loc[common_idx]
        target_aligned = target_volatility.loc[common_idx]
        
        # Select features
        if use_top_factors_only and hasattr(self, 'importance_model'):
            # Use top factors based on importance
            feature_importances = self.importance_model.feature_importances_
            top_indices = np.argsort(feature_importances)[-self.n_top_factors:]
            selected_factors = factors_aligned.iloc[:, top_indices]
        else:
            selected_factors = factors_aligned
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(selected_factors.values, target_aligned.values)
        
        # Evaluate
        predictions = model.predict(selected_factors.values)
        r2 = r2_score(target_aligned.values, predictions)
        rmse = np.sqrt(mean_squared_error(target_aligned.values, predictions))
        
        metrics = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': np.mean(np.abs(target_aligned.values - predictions)),
            'correlation': np.corrcoef(target_aligned.values, predictions)[0, 1]
        }
        
        return model, metrics


