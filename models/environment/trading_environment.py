"""
Trading Environment: SDE-Controlled State Transitions
====================================================

Quantitative Finance: Market State Evolution via SDE

Key Design Principle:
    State transition probability is determined by market SDE and volatility dynamics
    (Heston/GARCH), NOT by agent actions.

    Agent actions only affect portfolio (positions, cash, PnL), not market state transitions.

Mathematical Foundation:
    Market State Evolution (SDE-controlled):
        dS_t = μ_t S_t dt + σ_t S_t dW_t^(P)    [Real-world measure P]
        dV_t = κ(θ - V_t)dt + ξ√V_t dW_t^Q      [Heston volatility]
        
    Portfolio Evolution (Agent-controlled):
        dV_portfolio = Σ(π_i · dS_i) - costs
    
    Key Point: dS_t is independent of agent actions π_t
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ..options_pricing.heston import HestonModel
from ..alpha_signals import AlphaSignals, GARCHSignals


@dataclass
class MarketState:
    """
    Market state - evolves according to SDE, independent of agent actions.
    
    This state is controlled by:
    - SDE: dS_t = μ_t S_t dt + σ_t S_t dW_t
    - Volatility dynamics: Heston or GARCH
    """
    # Underlying prices (SDE-controlled)
    underlying_price: float
    volatility: float              # Current volatility (from Heston/GARCH)
    
    # Time
    time: float
    time_step: float = 1/252       # Daily time step
    
    # Market parameters (fixed, not affected by agents)
    risk_free_rate: float = 0.05
    drift: float = 0.0             # Real-world drift μ_t
    
    # Volatility state (Heston-specific)
    variance: Optional[float] = None  # V_t in Heston model
    
    # Metadata
    step: int = 0


@dataclass
class PortfolioState:
    """
    Portfolio state - evolves according to agent actions.
    
    This is the ONLY part affected by agent actions.
    """
    # Positions
    positions: Dict[str, float] = field(default_factory=dict)  # {instrument: quantity}
    
    # Cash
    cash: float = 1000000.0
    
    # Portfolio value
    portfolio_value: float = 1000000.0
    
    # Greeks (from current positions)
    greeks: Dict[str, float] = field(default_factory=dict)  # {delta, gamma, vega, ...}
    
    # Performance
    pnl: float = 0.0
    pnl_history: List[float] = field(default_factory=list)
    
    # Risk metrics
    current_risk: float = 0.0


class TradingEnvironment:
    """
    Trading Environment with SDE-Controlled State Transitions
    
    Design Principles:
    1. Market state evolves according to SDE (independent of agent actions)
    2. GARCH provides prediction signals to agents
    3. PDE provides theoretical prices and Greeks
    4. Agent actions ONLY affect portfolio, NOT market state transitions
    
    State Transition:
        P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t)  [Independent of action a_t]
        
    This is because market state follows SDE:
        dS_t = μ_t S_t dt + σ_t S_t dW_t
        which is independent of agent actions.
    """
    
    def __init__(self,
                 initial_price: float = 100.0,
                 initial_volatility: float = 0.2,
                 risk_free_rate: float = 0.05,
                 use_heston: bool = True,
                 heston_params: Optional[Dict[str, float]] = None):
        """
        Initialize trading environment.
        
        Parameters:
        -----------
        initial_price : float
            Initial underlying price
        initial_volatility : float
            Initial volatility
        risk_free_rate : float
            Risk-free rate
        use_heston : bool
            Whether to use Heston model for volatility dynamics
        heston_params : Dict, optional
            Heston model parameters
        """
        # Market state (SDE-controlled)
        self.market_state = MarketState(
            underlying_price=initial_price,
            volatility=initial_volatility,
            time=0.0,
            risk_free_rate=risk_free_rate
        )
        
        # Portfolio state (agent-controlled)
        self.portfolio_state = PortfolioState()
        
        # SDE parameters
        self.use_heston = use_heston
        if use_heston:
            heston_params = heston_params or {
                'v0': initial_volatility ** 2,
                'kappa': 2.0,
                'theta': 0.04,
                'sigma': 0.3,
                'rho': -0.7
            }
            self.heston_model = HestonModel(
                S0=initial_price,
                K=100.0,
                T=1.0,
                r=risk_free_rate,
                **heston_params
            )
            self.market_state.variance = heston_params['v0']
        else:
            self.heston_model = None
        
        # GARCH model (for prediction signals)
        self.garch_model = None
        
        # Pricing engine
        self.pricing_engine = None
        
        # Random seed for reproducibility
        self.rng = np.random.RandomState(42)
    
    def step(self, actions: Dict[str, Any]) -> Tuple[MarketState, Dict[str, float], bool]:
        """
        Execute one step in the environment.
        
        IMPORTANT: Market state transition is SDE-controlled and independent of actions.
        Only portfolio state is affected by agent actions.
        
        Parameters:
        -----------
        actions : Dict[str, Any]
            Actions from agents:
            - 'alpha': Trading action
            - 'market_maker': Quote action
            - 'hedging': Hedging action
        
        Returns:
        --------
        Tuple:
            - next_market_state: MarketState (SDE-evolved, independent of actions)
            - rewards: Dict[str, float] (rewards for each agent)
            - done: bool (episode termination)
        """
        # ========== STEP 1: Market State Transition (SDE-controlled) ==========
        # This is INDEPENDENT of agent actions
        next_market_state = self._evolve_market_state_sde()
        
        # ========== STEP 2: Apply Agent Actions (affect portfolio only) ==========
        # Agent actions only affect portfolio, not market state
        rewards = self._apply_actions(actions, next_market_state)
        
        # ========== STEP 3: Update Portfolio State ==========
        self._update_portfolio_state(next_market_state)
        
        # ========== STEP 4: Check Termination ==========
        done = self._check_termination()
        
        # Update market state
        self.market_state = next_market_state
        
        return next_market_state, rewards, done
    
    def _evolve_market_state_sde(self) -> MarketState:
        """
        Evolve market state according to SDE.
        
        This is the KEY method: market state evolves independently of agent actions.
        
        SDE Evolution:
            dS_t = μ_t S_t dt + σ_t S_t dW_t^(P)    [Real-world measure]
            
        If using Heston:
            dV_t = κ(θ - V_t)dt + ξ√V_t dW_t^Q
            σ_t = √V_t
        """
        dt = self.market_state.time_step
        
        if self.use_heston and self.market_state.variance is not None:
            # Heston volatility dynamics
            v_t = self.market_state.variance
            kappa = 2.0
            theta = 0.04
            xi = 0.3
            rho = -0.7
            
            # Evolve variance: dV_t = κ(θ - V_t)dt + ξ√V_t dW_2
            dW_2 = self.rng.normal(0, np.sqrt(dt))
            dv = kappa * (theta - v_t) * dt + xi * np.sqrt(max(v_t, 0)) * dW_2
            v_next = max(v_t + dv, 0.001)  # Ensure positive variance
            
            # Evolve price: dS_t = rS_t dt + √V_t S_t dW_1
            # Correlated Brownian motions: dW_1 = ρ·dW_2 + √(1-ρ²)·dW_3
            dW_3 = self.rng.normal(0, np.sqrt(dt))
            dW_1 = rho * dW_2 + np.sqrt(1 - rho**2) * dW_3
            
            dS = self.market_state.risk_free_rate * self.market_state.underlying_price * dt + \
                 np.sqrt(v_next) * self.market_state.underlying_price * dW_1
            
            S_next = max(self.market_state.underlying_price + dS, 0.01)
            vol_next = np.sqrt(v_next)
            
            next_state = MarketState(
                underlying_price=S_next,
                volatility=vol_next,
                variance=v_next,
                time=self.market_state.time + dt,
                time_step=self.market_state.time_step,
                risk_free_rate=self.market_state.risk_free_rate,
                drift=self.market_state.risk_free_rate,  # Risk-neutral drift
                step=self.market_state.step + 1
            )
        else:
            # Simple GBM: dS_t = μS_t dt + σS_t dW_t
            mu = self.market_state.drift if self.market_state.drift != 0 else self.market_state.risk_free_rate
            sigma = self.market_state.volatility
            
            dW = self.rng.normal(0, np.sqrt(dt))
            dS = mu * self.market_state.underlying_price * dt + \
                 sigma * self.market_state.underlying_price * dW
            
            S_next = max(self.market_state.underlying_price + dS, 0.01)
            
            # Volatility evolves (simplified, could use GARCH)
            vol_next = sigma  # Constant for now, could add GARCH evolution
            
            next_state = MarketState(
                underlying_price=S_next,
                volatility=vol_next,
                time=self.market_state.time + dt,
                time_step=self.market_state.time_step,
                risk_free_rate=self.market_state.risk_free_rate,
                drift=mu,
                step=self.market_state.step + 1
            )
        
        return next_state
    
    def _apply_actions(self, actions: Dict[str, Any], market_state: MarketState) -> Dict[str, float]:
        """
        Apply agent actions to portfolio.
        
        IMPORTANT: Actions only affect portfolio, NOT market state.
        Market state has already been evolved by SDE in _evolve_market_state_sde().
        
        Parameters:
        -----------
        actions : Dict[str, Any]
            Agent actions
        market_state : MarketState
            Current market state (already evolved by SDE)
        
        Returns:
        --------
        Dict[str, float]: Rewards for each agent
        """
        rewards = {}
        
        # Alpha agent action (affects portfolio positions)
        if 'alpha' in actions:
            alpha_action = actions['alpha']
            # Update portfolio positions based on action
            # This only affects portfolio_state, not market_state
            reward = self._apply_alpha_action(alpha_action, market_state)
            rewards['alpha'] = reward
        
        # Market maker action (affects portfolio via quotes)
        if 'market_maker' in actions:
            mm_action = actions['market_maker']
            # Market maker provides liquidity, affects portfolio
            reward = self._apply_market_maker_action(mm_action, market_state)
            rewards['market_maker'] = reward
        
        # Hedging agent action (affects portfolio via hedging)
        if 'hedging' in actions:
            hedge_action = actions['hedging']
            # Hedging affects portfolio risk, not market state
            reward = self._apply_hedging_action(hedge_action, market_state)
            rewards['hedging'] = reward
        
        return rewards
    
    def _apply_alpha_action(self, action, market_state: MarketState) -> float:
        """Apply alpha agent action (affects portfolio only)."""
        # Simplified: update portfolio positions
        # In full implementation, would execute trades and update positions
        return 0.0  # Placeholder reward
    
    def _apply_market_maker_action(self, action, market_state: MarketState) -> float:
        """Apply market maker action (affects portfolio only)."""
        # Market maker quotes affect portfolio via spread capture
        # Market state is NOT affected
        return 0.0  # Placeholder reward
    
    def _apply_hedging_action(self, action, market_state: MarketState) -> float:
        """Apply hedging action (affects portfolio only)."""
        # Hedging affects portfolio risk, not market state
        return 0.0  # Placeholder reward
    
    def _update_portfolio_state(self, market_state: MarketState) -> None:
        """Update portfolio state based on market state evolution."""
        # Mark-to-market portfolio
        # Update Greeks
        # Calculate PnL
        # This is where portfolio evolves based on market state changes
        pass
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Termination conditions:
        # - Time limit reached
        # - Portfolio value too low
        # - Maximum steps reached
        return self.market_state.step >= 252  # One year of trading days
    
    def reset(self) -> MarketState:
        """Reset environment to initial state."""
        self.market_state = MarketState(
            underlying_price=100.0,
            volatility=0.2,
            time=0.0,
            step=0
        )
        self.portfolio_state = PortfolioState()
        return self.market_state
    
    def get_alpha_signals(self) -> Optional[AlphaSignals]:
        """Get alpha signals (GARCH provides prediction signals)."""
        # This would use GARCH model to generate signals
        # Signals are predictions, not market state
        return None
    
    def get_pricing_info(self) -> Dict[str, Any]:
        """
        Get pricing information (PDE provides theoretical prices and Greeks).
        
        Returns:
        --------
        Dict containing:
            - theoretical_price: float (from PDE)
            - greeks: Dict[str, float] (Delta, Gamma, Vega, ...)
            - garch_signals: Optional[GARCHSignals]
        """
        if self.pricing_engine:
            theoretical_price = self.pricing_engine.price()
            greeks = self.pricing_engine.greeks()
        else:
            theoretical_price = self.market_state.underlying_price
            greeks = {}
        
        return {
            'theoretical_price': theoretical_price,
            'greeks': greeks,
            'garch_signals': None  # Would be populated from GARCH model
        }
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get portfolio state for hedging agent."""
        return {
            'greeks': self.portfolio_state.greeks,
            'value': self.portfolio_state.portfolio_value,
            'ou_signals': None,  # Would be populated from OU model
            'garch_signals': None,  # Would be populated from GARCH model
            'copula_signals': None  # Would be populated from Copula model
        }


# ========== Key Design Document ==========

"""
DESIGN PRINCIPLES: SDE-Controlled State Transitions

1. Market State Evolution (SDE-Controlled):
   ----------------------------------------
   Market state evolves according to SDE:
       dS_t = μ_t S_t dt + σ_t S_t dW_t^(P)
       dV_t = κ(θ - V_t)dt + ξ√V_t dW_t^Q  [Heston]
   
   This evolution is:
   ✓ Independent of agent actions
   ✓ Determined by SDE parameters and random shocks
   ✓ Represents "true" market dynamics
   
   State Transition Probability:
       P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t)
       
   Key Point: Action a_t does NOT appear in the transition probability!

2. Agent Actions (Portfolio-Only):
   --------------------------------
   Agent actions ONLY affect:
   ✓ Portfolio positions
   ✓ Portfolio cash
   ✓ Portfolio PnL
   ✓ Portfolio Greeks
   
   Agent actions do NOT affect:
   ✗ Market price evolution (S_t)
   ✗ Volatility evolution (σ_t or V_t)
   ✗ Market state transition probabilities
   
3. GARCH Provides Prediction Signals:
   -----------------------------------
   GARCH model provides:
   ✓ Volatility forecasts (σ̂_t)
   ✓ Prediction signals for agents
   
   But GARCH does NOT control market state evolution.
   Market state still evolves according to SDE.

4. PDE Provides Theoretical Prices:
   --------------------------------
   PDE/SDE pricing models provide:
   ✓ Theoretical option prices
   ✓ Greeks (Δ, Γ, Vega, Θ, ρ)
   ✓ Risk metrics
   
   These are used by agents for decision-making, but do NOT affect
   market state evolution.

5. Implementation:
   ---------------
   In step() method:
   1. First: Evolve market state via SDE (_evolve_market_state_sde)
   2. Then: Apply agent actions to portfolio (_apply_actions)
   3. Finally: Update portfolio state (_update_portfolio_state)
   
   This ensures market state evolution is independent of agent actions.
"""



