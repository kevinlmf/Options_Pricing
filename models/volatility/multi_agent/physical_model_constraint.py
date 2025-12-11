"""
Physical Model Constraint for Multi-Agent Volatility
====================================================

Integrates multi-agent factor extraction with traditional physical models
(jump-diffusion, stochastic volatility) to ensure convergence and provide
theoretical grounding.

Key Idea:
- Multi-agent provides micro-mechanism: agent behavior → factors → volatility
- Physical models provide macro-constraint: ensure convergence to known distributions
- Time-series modeling connects micro and macro: factor dynamics → physical model parameters

This ensures:
1. Better volatility prediction through agent behavior understanding
2. Theoretical convergence guarantees through physical model constraints
3. Time-series modeling bridges micro and macro scales
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats, optimize
from scipy.stats import norm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@dataclass
class JumpDiffusionParams:
    """Parameters for jump-diffusion model"""
    mu: float          # Drift
    sigma: float       # Diffusion volatility
    lambda_jump: float # Jump intensity
    mu_jump: float     # Mean jump size
    sigma_jump: float  # Jump volatility


@dataclass
class PhysicalModelFit:
    """Physical model fitting results"""
    model_type: str  # 'jump_diffusion', 'heston', etc.
    parameters: Dict[str, float]
    log_likelihood: float
    convergence_metric: float  # How well it matches long-term distribution
    factor_loadings: Dict[str, float]  # How factors map to model parameters


class JumpDiffusionModel:
    """
    Jump-Diffusion Model for volatility.
    
    dS_t = μ S_t dt + σ S_t dW_t + J_t dN_t
    
    Where:
    - dW_t: Brownian motion (diffusion)
    - dN_t: Poisson process (jumps)
    - J_t: Jump size (normal distribution)
    """
    
    def __init__(self):
        self.params: Optional[JumpDiffusionParams] = None
    
    def fit(self, returns: np.ndarray) -> JumpDiffusionParams:
        """
        Fit jump-diffusion model to returns using MLE.
        
        Parameters:
        -----------
        returns : np.ndarray
            Asset returns
            
        Returns:
        --------
        JumpDiffusionParams : Fitted parameters
        """
        # Initial parameter estimates
        mu_init = np.mean(returns)
        sigma_init = np.std(returns)
        lambda_init = 0.1  # Low jump frequency
        mu_jump_init = 0.0
        sigma_jump_init = sigma_init * 0.5
        
        def negative_log_likelihood(params):
            mu, sigma, lambda_jump, mu_jump, sigma_jump = params
            
            # Ensure positive parameters
            if sigma <= 0 or lambda_jump < 0 or sigma_jump <= 0:
                return 1e10
            
            # Calculate log-likelihood
            # Simplified: mixture of normal (diffusion) and normal (jump)
            ll = 0.0
            for r in returns:
                # Diffusion component
                ll_diffusion = norm.logpdf(r, mu, sigma)
                
                # Jump component (simplified)
                ll_jump = norm.logpdf(r, mu + mu_jump, np.sqrt(sigma**2 + sigma_jump**2))
                
                # Mixture: (1 - lambda) * diffusion + lambda * jump
                ll += np.log((1 - lambda_jump) * np.exp(ll_diffusion) + 
                            lambda_jump * np.exp(ll_jump))
            
            return -ll  # Negative for minimization
        
        # Optimize
        result = optimize.minimize(
            negative_log_likelihood,
            x0=[mu_init, sigma_init, lambda_init, mu_jump_init, sigma_jump_init],
            method='L-BFGS-B',
            bounds=[(None, None), (1e-6, None), (0, 1), (None, None), (1e-6, None)]
        )
        
        self.params = JumpDiffusionParams(
            mu=result.x[0],
            sigma=result.x[1],
            lambda_jump=result.x[2],
            mu_jump=result.x[3],
            sigma_jump=result.x[4]
        )
        
        return self.params
    
    def simulate(self, n_steps: int, dt: float = 1/252) -> np.ndarray:
        """
        Simulate jump-diffusion process.
        
        Parameters:
        -----------
        n_steps : int
            Number of steps
        dt : float
            Time step
            
        Returns:
        --------
        np.ndarray : Simulated returns
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        returns = np.zeros(n_steps)
        
        for t in range(n_steps):
            # Diffusion component
            diffusion = self.params.mu * dt + self.params.sigma * np.sqrt(dt) * np.random.randn()
            
            # Jump component
            n_jumps = np.random.poisson(self.params.lambda_jump * dt)
            jump = 0.0
            if n_jumps > 0:
                jump = np.sum(np.random.normal(self.params.mu_jump, self.params.sigma_jump, n_jumps))
            
            returns[t] = diffusion + jump
        
        return returns
    
    def long_term_volatility(self) -> float:
        """
        Calculate long-term volatility (convergence target).
        
        For jump-diffusion:
        σ_long = sqrt(σ² + λ * (μ_jump² + σ_jump²))
        """
        if self.params is None:
            raise ValueError("Model not fitted.")
        
        diffusion_var = self.params.sigma ** 2
        jump_var = self.params.lambda_jump * (
            self.params.mu_jump ** 2 + self.params.sigma_jump ** 2
        )
        
        return np.sqrt(diffusion_var + jump_var)


class PhysicalModelConstraint:
    """
    Constrains multi-agent factors to converge to physical model predictions.
    
    Approach:
    1. Fit physical model (jump-diffusion) to historical data
    2. Extract long-term volatility target from physical model
    3. Use time-series modeling to ensure agent factors converge to this target
    4. Factor dynamics: dF_t = κ(θ - F_t)dt + σ_F dW_t
       where θ is the physical model target
    """
    
    def __init__(self, 
                 physical_model: str = 'jump_diffusion',
                 convergence_rate: float = 0.1):
        """
        Initialize constraint.
        
        Parameters:
        -----------
        physical_model : str
            Type of physical model ('jump_diffusion', 'heston', etc.)
        convergence_rate : float
            Rate of convergence (κ in mean-reversion)
        """
        self.physical_model_type = physical_model
        self.convergence_rate = convergence_rate
        
        if physical_model == 'jump_diffusion':
            self.physical_model = JumpDiffusionModel()
        else:
            raise ValueError(f"Unknown physical model: {physical_model}")
        
        self.fitted_model: Optional[PhysicalModelFit] = None
    
    def fit_physical_model(self, returns: np.ndarray) -> PhysicalModelFit:
        """
        Fit physical model to returns.
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns
            
        Returns:
        --------
        PhysicalModelFit : Fitted model results
        """
        # Fit physical model
        params = self.physical_model.fit(returns)
        
        # Calculate long-term volatility (convergence target)
        long_term_vol = self.physical_model.long_term_volatility()
        
        # Calculate log-likelihood
        log_likelihood = self._calculate_log_likelihood(returns, params)
        
        self.fitted_model = PhysicalModelFit(
            model_type=self.physical_model_type,
            parameters={
                'mu': params.mu,
                'sigma': params.sigma,
                'lambda_jump': params.lambda_jump,
                'mu_jump': params.mu_jump,
                'sigma_jump': params.sigma_jump,
                'long_term_volatility': long_term_vol
            },
            log_likelihood=log_likelihood,
            convergence_metric=long_term_vol,
            factor_loadings={}  # Will be filled later
        )
        
        return self.fitted_model
    
    def _calculate_log_likelihood(self, returns: np.ndarray, params: JumpDiffusionParams) -> float:
        """Calculate log-likelihood for fitted model"""
        ll = 0.0
        for r in returns:
            ll_diffusion = norm.logpdf(r, params.mu, params.sigma)
            ll_jump = norm.logpdf(r, params.mu + params.mu_jump, 
                                 np.sqrt(params.sigma**2 + params.sigma_jump**2))
            ll += np.log((1 - params.lambda_jump) * np.exp(ll_diffusion) + 
                        params.lambda_jump * np.exp(ll_jump))
        return ll
    
    def map_factors_to_physical_model(self,
                                     factors: pd.DataFrame,
                                     target_volatility: pd.Series) -> Dict[str, float]:
        """
        Map multi-agent factors to physical model parameters.
        
        Uses time-series regression to find how factors contribute to
        physical model parameters.
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Multi-agent factors
        target_volatility : pd.Series
            Target volatility (from physical model or realized)
            
        Returns:
        --------
        Dict[str, float] : Factor loadings on physical model parameters
        """
        # Align data
        common_idx = factors.index.intersection(target_volatility.index)
        factors_aligned = factors.loc[common_idx]
        target_aligned = target_volatility.loc[common_idx]
        
        if len(common_idx) < 10:
            return {}
        
        # Time-series regression: factors → target volatility
        # This connects micro (agent factors) to macro (physical model)

        # Ensure no NaN values
        if target_aligned.isna().any():
            print(f"Warning: Target contains NaN values, dropping them")
            valid_mask = ~target_aligned.isna()
            factors_aligned = factors_aligned.loc[valid_mask]
            target_aligned = target_aligned.loc[valid_mask]

        if len(factors_aligned) < 5:
            print(f"Warning: Not enough data after NaN removal ({len(factors_aligned)} points)")
            return {}

        from sklearn.linear_model import LinearRegression

        try:
            model = LinearRegression()
            model.fit(factors_aligned.values, target_aligned.values)

            factor_loadings = {
                factor_name: coef
                for factor_name, coef in zip(factors_aligned.columns, model.coef_)
            }
        except Exception as e:
            print(f"Warning: Regression failed: {e}")
            # Return zero loadings as fallback
            factor_loadings = {col: 0.0 for col in factors_aligned.columns}
        
        # Update fitted model
        if self.fitted_model:
            self.fitted_model.factor_loadings = factor_loadings
        
        return factor_loadings
    
    def constrained_factor_dynamics(self,
                                  factors: pd.DataFrame,
                                  physical_target: float,
                                  realized_volatility: pd.Series = None) -> pd.DataFrame:
        """
        Apply dual constraint: match realized volatility (short-term) + converge to physical model (long-term).

        Factor dynamics with dual constraints:
        dF_t = κ₁(F_t - realized_vol_t)dt + κ₂(θ - F_t)dt + σ_F dW_t

        Where:
        - κ₁: Short-term adjustment to match realized volatility
        - κ₂: Long-term convergence to physical model target θ
        - realized_vol_t: Actual market volatility (short-term target)
        - θ: Physical model long-term volatility target

        This ensures:
        1. Short-term: Factors track actual volatility patterns
        2. Long-term: Factors converge to theoretical volatility

        Parameters:
        -----------
        factors : pd.DataFrame
            Current factor values
        physical_target : float
            Long-term volatility from physical model
        realized_volatility : pd.Series, optional
            Realized volatility for short-term matching

        Returns:
        --------
        pd.DataFrame : Constrained factors (dual convergence)
        """
        constrained_factors = factors.copy()

        for col in factors.columns:
            current_factor = factors[col].values

            # Long-term convergence to physical model
            long_term_adjustment = self.convergence_rate * (physical_target - current_factor)

            # Short-term adjustment to match realized volatility (if provided)
            short_term_adjustment = np.zeros_like(current_factor)
            if realized_volatility is not None:
                # Align indices
                common_idx = factors.index.intersection(realized_volatility.index)
                if len(common_idx) > 0:
                    realized_aligned = realized_volatility.loc[common_idx]
                    factor_aligned = factors.loc[common_idx, col]

                    # Short-term adjustment: match realized vol with smaller rate
                    short_rate = self.convergence_rate * 0.3  # Smaller rate for short-term
                    short_term_adjustment = short_rate * (realized_aligned.values - factor_aligned.values)

                    # Apply short-term adjustment
                    constrained_factors.loc[common_idx, col] = (
                        factor_aligned.values + short_term_adjustment
                    )

            # Apply long-term adjustment
            constrained_factors[col] = constrained_factors[col] + long_term_adjustment

        return constrained_factors
    
    def check_convergence(self,
                         factors: pd.DataFrame,
                         window: int = 20) -> Dict[str, float]:
        """
        Check if factors are converging to physical model target.
        
        Parameters:
        -----------
        factors : pd.DataFrame
            Factor time series
        window : int
            Rolling window for convergence check
            
        Returns:
        --------
        Dict[str, float] : Convergence metrics for each factor
        """
        if self.fitted_model is None:
            return {}
        
        target = self.fitted_model.parameters.get('long_term_volatility', 0.2)
        convergence_metrics = {}
        
        for col in factors.columns:
            recent_factors = factors[col].rolling(window).mean()
            distance_to_target = abs(recent_factors.iloc[-1] - target) / target
            convergence_metrics[col] = 1.0 - min(distance_to_target, 1.0)  # 0-1 scale
        
        return convergence_metrics

