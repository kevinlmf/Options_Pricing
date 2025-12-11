"""
Multi-Agent Volatility Extractor
=================================

Extract volatility from agent interactions and emergent market behavior.

Core Concept:
------------
Volatility is NOT an exogenous parameter.
Volatility EMERGES from agent interactions.

Agent Types and Volatility Signals:
-----------------------------------
1. Market Maker: Bid-ask spread → Market uncertainty
2. Arbitrageur: Trading frequency → Market efficiency  
3. Trend Follower: Herding ratio → Volatility amplification
4. Fundamental Investor: Holdings → Market stability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@dataclass
class AgentState:
    """State of an agent at a given time"""
    agent_type: str
    position: float
    cash: float
    last_action: str
    confidence: float


@dataclass
class MarketState:
    """Overall market state"""
    price: float
    volume: int
    bid_ask_spread: float
    num_trades: int
    timestamp: int


class MarketMakerAgent:
    """
    Market Maker Agent
    
    Behavior:
    - Provides liquidity by quoting bid/ask prices
    - Adjusts spread based on perceived risk
    
    Volatility Signal:
    - Wider spread → Higher uncertainty → Higher volatility
    """
    
    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion
        self.inventory = 0
        self.cash = 100000
        self.base_spread = 0.001  # 10 bps
        
    def quote_spread(self, market_state: MarketState, recent_volatility: float) -> float:
        """
        Determine bid-ask spread based on market conditions
        
        Spread increases with:
        - Recent volatility
        - Inventory risk
        - Trading volume (liquidity demand)
        """
        # Base spread
        spread = self.base_spread
        
        # Adjust for volatility
        spread *= (1 + recent_volatility * 10)
        
        # Adjust for inventory risk
        inventory_risk = abs(self.inventory) / 1000
        spread *= (1 + inventory_risk * self.risk_aversion)
        
        # Adjust for volume (liquidity demand)
        if market_state.volume > 0:
            volume_factor = market_state.volume / 10000
            spread *= (1 + volume_factor * 0.5)
        
        return spread
    
    def extract_volatility_signal(self, spread: float) -> float:
        """
        Convert spread to volatility signal
        
        Logic: Spread reflects market maker's uncertainty
        """
        # Normalize spread to volatility (annualized)
        # Typical spread: 10-50 bps → volatility: 10-30%
        volatility = spread * 20  # Calibration factor
        return min(volatility, 1.0)  # Cap at 100%


class ArbitrageurAgent:
    """
    Arbitrageur Agent
    
    Behavior:
    - Seeks mispricing opportunities
    - Trades frequently when opportunities exist
    
    Volatility Signal:
    - High trading frequency → Market inefficiency → Higher volatility
    - Low trading frequency → Efficient market → Lower volatility
    """
    
    def __init__(self, threshold: float = 0.005):
        self.threshold = threshold  # Minimum profit threshold
        self.trade_count = 0
        self.cash = 100000
        
    def detect_arbitrage(self, market_price: float, fair_value: float) -> bool:
        """
        Detect if arbitrage opportunity exists
        """
        mispricing = abs(market_price - fair_value) / fair_value
        return mispricing > self.threshold
    
    def extract_volatility_signal(self, trade_frequency: float, max_frequency: float = 100) -> float:
        """
        Convert trading frequency to volatility signal
        
        Logic: More arbitrage opportunities → More mispricing → Higher volatility
        """
        # Normalize frequency to [0, 1]
        normalized_frequency = min(trade_frequency / max_frequency, 1.0)
        
        # Map to volatility (10% to 50%)
        volatility = 0.10 + normalized_frequency * 0.40
        return volatility


class TrendFollowerAgent:
    """
    Trend Follower Agent
    
    Behavior:
    - Buys when price is rising
    - Sells when price is falling
    - Creates momentum and herding
    
    Volatility Signal:
    - High herding ratio → Volatility amplification
    - Positive feedback loop
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.position = 0
        self.cash = 100000
        
    def detect_trend(self, price_history: List[float]) -> str:
        """
        Detect trend direction
        """
        if len(price_history) < self.lookback:
            return 'neutral'
        
        recent_prices = price_history[-self.lookback:]
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if trend > 0.02:
            return 'up'
        elif trend < -0.02:
            return 'down'
        else:
            return 'neutral'
    
    def extract_volatility_signal(self, herding_ratio: float) -> float:
        """
        Convert herding ratio to volatility signal
        
        Logic: More herding → Stronger momentum → Higher volatility
        
        Parameters:
        -----------
        herding_ratio : float
            Proportion of trend followers in the market (0 to 1)
        """
        # Herding amplifies volatility
        # Base volatility: 15%
        # Max amplification: 2x
        base_vol = 0.15
        amplification = 1 + herding_ratio
        
        return base_vol * amplification


class FundamentalInvestorAgent:
    """
    Fundamental Investor Agent
    
    Behavior:
    - Estimates intrinsic value
    - Buys when undervalued, sells when overvalued
    - Long-term horizon, stabilizing force
    
    Volatility Signal:
    - High fundamental holdings → Market stability → Lower volatility
    - Low fundamental holdings → Speculation dominates → Higher volatility
    """
    
    def __init__(self, valuation_model='DCF'):
        self.valuation_model = valuation_model
        self.holdings = 0
        self.cash = 100000
        
    def estimate_intrinsic_value(self, market_data: Dict) -> float:
        """
        Estimate intrinsic value using fundamental analysis
        """
        # Simplified valuation
        # In practice, would use DCF, multiples, etc.
        current_price = market_data.get('price', 100)
        
        # Add some noise to simulate valuation uncertainty
        intrinsic_value = current_price * (1 + np.random.normal(0, 0.1))
        return intrinsic_value
    
    def extract_volatility_signal(self, fundamental_ratio: float) -> float:
        """
        Convert fundamental investor ratio to volatility signal
        
        Logic: More fundamental investors → More stability → Lower volatility
        
        Parameters:
        -----------
        fundamental_ratio : float
            Proportion of fundamental investors (0 to 1)
        """
        # Fundamental investors stabilize the market
        # Base volatility: 20%
        # Max stabilization: 0.5x
        base_vol = 0.20
        stabilization = 1 - (fundamental_ratio * 0.5)
        
        return base_vol * stabilization


class MultiAgentVolatilityExtractor:
    """
    Extract volatility from multi-agent market simulation
    
    Combines signals from all agent types to estimate market volatility
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.market_maker = MarketMakerAgent()
        self.arbitrageur = ArbitrageurAgent()
        self.trend_follower = TrendFollowerAgent()
        self.fundamental_investor = FundamentalInvestorAgent()
        
        # Weights for combining signals
        if weights:
            self.weights = weights
        else:
            self.weights = {
                'market_maker': 0.30,
                'arbitrageur': 0.30,
                'trend_follower': 0.20,
                'fundamental': 0.20
            }
        
    def extract_volatility(self, 
                          market_state: MarketState,
                          agent_states: Dict[str, AgentState],
                          price_history: List[float]) -> Dict[str, float]:
        """
        Extract volatility from agent interactions
        
        Parameters:
        -----------
        market_state : MarketState
            Current market state
        agent_states : dict
            States of all agents
        price_history : list
            Historical prices
            
        Returns:
        --------
        dict : Volatility estimates from each agent type and combined
        """
        # Calculate recent realized volatility (for market maker)
        if len(price_history) > 20:
            returns = np.diff(np.log(price_history[-20:]))
            recent_vol = np.std(returns) * np.sqrt(252)
        else:
            recent_vol = 0.15
        
        # 1. Market Maker signal (from spread)
        spread = self.market_maker.quote_spread(market_state, recent_vol)
        vol_market_maker = self.market_maker.extract_volatility_signal(spread)
        
        # 2. Arbitrageur signal (from trading frequency)
        arbitrage_count = sum(1 for state in agent_states.values() 
                            if state.agent_type == 'arbitrageur' and state.last_action != 'hold')
        vol_arbitrageur = self.arbitrageur.extract_volatility_signal(arbitrage_count)
        
        # 3. Trend Follower signal (from herding ratio)
        total_agents = len(agent_states)
        trend_followers = sum(1 for state in agent_states.values() 
                            if state.agent_type == 'trend_follower')
        herding_ratio = trend_followers / total_agents if total_agents > 0 else 0
        vol_trend_follower = self.trend_follower.extract_volatility_signal(herding_ratio)
        
        # 4. Fundamental Investor signal (from fundamental ratio)
        fundamental_investors = sum(1 for state in agent_states.values() 
                                   if state.agent_type == 'fundamental')
        fundamental_ratio = fundamental_investors / total_agents if total_agents > 0 else 0
        vol_fundamental = self.fundamental_investor.extract_volatility_signal(fundamental_ratio)
        
        # Combine signals
        combined_volatility = (
            self.weights['market_maker'] * vol_market_maker +
            self.weights['arbitrageur'] * vol_arbitrageur +
            self.weights['trend_follower'] * vol_trend_follower +
            self.weights['fundamental'] * vol_fundamental
        )
        
        return {
            'market_maker_vol': vol_market_maker,
            'arbitrageur_vol': vol_arbitrageur,
            'trend_follower_vol': vol_trend_follower,
            'fundamental_vol': vol_fundamental,
            'combined_volatility': combined_volatility,
            'weights': self.weights
        }
    
    def simulate_and_extract(self, 
                            initial_price: float = 100,
                            n_steps: int = 100,
                            n_agents: int = 100,
                            external_price_history: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Run multi-agent simulation and extract volatility over time
        
        Returns:
        --------
        pd.DataFrame : Time series of volatility estimates
        """
        # Initialize agents
        agent_types = ['market_maker', 'arbitrageur', 'trend_follower', 'fundamental']
        agents = []
        
        for i in range(n_agents):
            agent_type = agent_types[i % len(agent_types)]
            agents.append(AgentState(
                agent_type=agent_type,
                position=0,
                cash=100000,
                last_action='hold',
                confidence=0.5
            ))
        
        # Simulation
        if external_price_history:
            price_history = external_price_history
            n_steps = len(price_history) - 1
            # If using external history, we iterate through it
        else:
            price_history = [initial_price]
        
        volatility_history = []
        
        # Intrinsic Value Process (Random Walk for Fundamental Reference)
        intrinsic_value = initial_price
        
        for t in range(n_steps):
            market_state_prev = MarketState(
                price=price_history[-1],
                volume=0, bid_ask_spread=0.01, num_trades=0, timestamp=t
            )
            
            # Agent Decisions First (Actions determine price if no external history)
            # ------------------------------------------------------------------
            buy_pressure = 0
            sell_pressure = 0
            
            # Update Intrinsic Value (Slow drift)
            intrinsic_value *= (1 + np.random.normal(0, 0.005))
            
            agent_states = {}
            for i, agent in enumerate(agents):
                # Determine action
                action = 'hold'
                
                if agent.agent_type == 'trend_follower':
                    if len(price_history) > 5:
                        trend = price_history[-1] - price_history[-5]
                        if trend > 0.05: action = 'buy'
                        elif trend < -0.05: action = 'sell'
                
                elif agent.agent_type == 'fundamental':
                    # Compare price to intrinsic value
                    if price_history[-1] < intrinsic_value * 0.95:
                        action = 'buy'
                    elif price_history[-1] > intrinsic_value * 1.05:
                        action = 'sell'
                        
                elif agent.agent_type == 'arbitrageur':
                    if np.random.random() < 0.1: # Opportunistic
                         # simplified mean reversion
                         if price_history[-1] > np.mean(price_history[-20:]) if len(price_history)>20 else False:
                             action = 'sell'
                         else:
                             action = 'buy'

                agent.last_action = action
                if action == 'buy': buy_pressure += 1
                elif action == 'sell': sell_pressure += 1
                
                agent_states[f'agent_{i}'] = agent

            if external_price_history:
                current_price = price_history[t]
                new_price = price_history[t+1]
                price_change = new_price - current_price
            else:
                # Agent-Driven Price Formation
                current_price = price_history[-1]
                
                # Net Imbalance
                net_imbalance = buy_pressure - sell_pressure
                
                # Liquidity factor (Market Makers dampen impact)
                # More market makers = less impact
                n_market_makers = sum(1 for a in agents if a.agent_type == 'market_maker')
                impact_factor = 0.01 / (1 + n_market_makers * 0.05)
                
                # Deterministic Impact + Noise
                price_impact = net_imbalance * impact_factor
                noise = np.random.normal(0, 0.005) # Market noise
                
                price_return = price_impact + noise
                new_price = current_price * (1 + price_return)
                price_history.append(new_price)
            
            # Market state update
            market_state = MarketState(
                price=new_price,
                volume=int((buy_pressure + sell_pressure) * 100),
                bid_ask_spread=self.market_maker.quote_spread(market_state_prev, 0.15), # approx
                num_trades=buy_pressure + sell_pressure,
                timestamp=t
            )
            
            # Extract volatility
            vol_signals = self.extract_volatility(market_state, agent_states, price_history)
            volatility_history.append(vol_signals)
        
        # Convert to DataFrame
        df = pd.DataFrame(volatility_history)
        if external_price_history:
             df['price'] = price_history[1:]
        else:
             df['price'] = price_history[1:]  # Exclude initial price
        
        return df


if __name__ == "__main__":
    print("Multi-Agent Volatility Extraction Demonstration")
    print("=" * 60)
    
    # Create extractor
    extractor = MultiAgentVolatilityExtractor()
    
    # Run simulation
    print("\nRunning multi-agent simulation (100 steps, 100 agents)...")
    results = extractor.simulate_and_extract(n_steps=100, n_agents=100)
    
    print(f"\nSimulation complete!")
    print(f"Price range: ${results['price'].min():.2f} - ${results['price'].max():.2f}")
    print(f"\nVolatility estimates:")
    print(f"  Market Maker:      {results['market_maker_vol'].mean():.2%} (avg)")
    print(f"  Arbitrageur:       {results['arbitrageur_vol'].mean():.2%} (avg)")
    print(f"  Trend Follower:    {results['trend_follower_vol'].mean():.2%} (avg)")
    print(f"  Fundamental:       {results['fundamental_vol'].mean():.2%} (avg)")
    print(f"  Combined:          {results['combined_volatility'].mean():.2%} (avg)")
    
    print(f"\nVolatility statistics:")
    print(results[['market_maker_vol', 'arbitrageur_vol', 'trend_follower_vol', 
                   'fundamental_vol', 'combined_volatility']].describe())
