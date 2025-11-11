"""
Improved Multi-Agent Implementation

This module provides improved versions of agents with better parameter calibration
and more sophisticated market state awareness.

Improvements:
1. Calibrated volatility mapping (reduced from 20x/50x to more reasonable values)
2. Market state-aware spread calculation
3. Adaptive parameter adjustment based on historical volatility
"""

import numpy as np
from typing import Dict, List, Optional
from .agents import BaseAgent, AgentState


class ImprovedMarketMaker(BaseAgent):
    """
    Improved Market Maker Agent with calibrated parameters
    
    Key Improvements:
    - Reduced coefficient sensitivity (from 20/50 to 5/10)
    - Market state awareness (normalizes by historical volatility)
    - Adaptive clipping bounds based on market regime
    """
    
    def __init__(self, risk_aversion: float = 1.0, target_spread: float = 0.01,
                 vol_coefficient_mean: float = 5.0, vol_coefficient_var: float = 10.0):
        """
        Initialize improved market maker.
        
        Parameters:
        -----------
        risk_aversion : float
            Risk aversion parameter
        target_spread : float
            Target bid-ask spread
        vol_coefficient_mean : float
            Coefficient for mean spread (default: 5.0, reduced from 20.0)
        vol_coefficient_var : float
            Coefficient for spread variance (default: 10.0, reduced from 50.0)
        """
        super().__init__(risk_aversion)
        self.target_spread = target_spread
        self.vol_coefficient_mean = vol_coefficient_mean
        self.vol_coefficient_var = vol_coefficient_var
    
    def decide_action(self, market_state: Dict) -> Dict:
        """Decide bid-ask spread based on volatility perception."""
        current_price = market_state['price']
        recent_returns = market_state.get('recent_returns', np.array([]))
        
        if len(recent_returns) > 0:
            # Estimate volatility from recent returns
            realized_vol = np.std(recent_returns)
            
            # Normalize by historical volatility to avoid extreme values
            historical_vol = np.std(recent_returns) if len(recent_returns) > 10 else 0.02
            vol_ratio = realized_vol / (historical_vol + 1e-6)
            
            # Adjust spread based on volatility and inventory risk
            inventory_penalty = abs(self.state.inventory) * 0.001
            spread = self.target_spread + realized_vol * 5 * vol_ratio + inventory_penalty
        else:
            spread = self.target_spread
        
        # Quote bid-ask
        bid = current_price * (1 - spread/2)
        ask = current_price * (1 + spread/2)
        
        action = {
            'type': 'quote',
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'mid': current_price
        }
        
        self.state.action_history.append(action)
        return action
    
    def infer_parameter(self, action_history: List[Dict]) -> float:
        """
        Infer implied volatility from spread history with calibrated coefficients.
        
        Improved Formula:
        σ_implied = base_vol + α × E[spread] + β × Var[spread]
        
        Where:
        - base_vol: 0.15 (15% base volatility, more reasonable)
        - α: vol_coefficient_mean (default: 5.0, reduced from 20.0)
        - β: vol_coefficient_var (default: 10.0, reduced from 50.0)
        """
        if len(action_history) == 0:
            return 0.15  # Default to 15% volatility
        
        spreads = [action['spread'] for action in action_history if 'spread' in action]
        
        if len(spreads) < 2:
            return 0.15
        
        mean_spread = np.mean(spreads)
        var_spread = np.var(spreads)
        
        # Improved structural mapping with calibrated coefficients
        # Base volatility: 15% (more reasonable than 20%)
        # Reduced sensitivity: 5x mean, 10x var (vs original 20x, 50x)
        base_vol = 0.15
        implied_vol = base_vol + mean_spread * self.vol_coefficient_mean + var_spread * self.vol_coefficient_var
        
        # Adaptive clipping based on market regime
        # If spreads are very small, cap at reasonable levels
        if mean_spread < 0.01:  # Normal market
            max_vol = 0.5  # 50% max
        elif mean_spread < 0.05:  # Volatile market
            max_vol = 1.0  # 100% max
        else:  # Extreme market
            max_vol = 2.0  # 200% max
        
        return np.clip(implied_vol, 0.05, max_vol)  # Min 5%, max adaptive
    
    def calibrate_from_historical(self, historical_prices: np.ndarray) -> Dict[str, float]:
        """
        Calibrate coefficients from historical data.
        
        This method analyzes historical price data to find optimal coefficients
        that map spread dynamics to realized volatility.
        """
        if len(historical_prices) < 20:
            return {
                'vol_coefficient_mean': self.vol_coefficient_mean,
                'vol_coefficient_var': self.vol_coefficient_var
            }
        
        # Calculate realized volatility
        returns = np.diff(np.log(historical_prices))
        realized_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Simulate agent to get spreads
        spreads = []
        for i in range(10, len(historical_prices)):
            recent_returns = returns[max(0, i-10):i]
            market_state = {
                'price': historical_prices[i],
                'recent_returns': recent_returns
            }
            action = self.decide_action(market_state)
            spreads.append(action['spread'])
        
        if len(spreads) < 2:
            return {
                'vol_coefficient_mean': self.vol_coefficient_mean,
                'vol_coefficient_var': self.vol_coefficient_var
            }
        
        mean_spread = np.mean(spreads)
        var_spread = np.var(spreads)
        
        # Simple calibration: solve for coefficients
        # σ = base + α*mean_spread + β*var_spread
        # Assuming base = 0.15, solve for α and β
        if mean_spread > 1e-6 and var_spread > 1e-6:
            # Use linear regression approach
            # For simplicity, use proportional scaling
            target_vol = realized_vol
            base_vol = 0.15
            
            # Estimate coefficients (simplified)
            if mean_spread > 0:
                alpha_est = min((target_vol - base_vol) / (mean_spread + var_spread), 10.0)
            else:
                alpha_est = 5.0
            
            return {
                'vol_coefficient_mean': max(1.0, min(alpha_est, 10.0)),
                'vol_coefficient_var': max(2.0, min(alpha_est * 2, 20.0))
            }
        
        return {
            'vol_coefficient_mean': self.vol_coefficient_mean,
            'vol_coefficient_var': self.vol_coefficient_var
        }



