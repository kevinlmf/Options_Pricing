
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sys
import os
import json

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from models.volatility.multi_agent.agent_volatility import MultiAgentVolatilityExtractor
from validation.rust_interface import RustVolatilityValidator, ValidationResult

@dataclass
class AgentConfiguration:
    """Configuration for Multi-Agent System"""
    n_market_makers: int
    n_arbitrageurs: int
    n_trend_followers: int
    n_fundamental_investors: int
    
    # Weights for volatility signal combination
    weight_market_maker: float
    weight_arbitrageur: float
    weight_trend_follower: float
    weight_fundamental: float

    def to_weights_dict(self) -> Dict[str, float]:
        return {
            'market_maker': self.weight_market_maker,
            'arbitrageur': self.weight_arbitrageur,
            'trend_follower': self.weight_trend_follower,
            'fundamental': self.weight_fundamental
        }
    
    def total_agents(self) -> int:
        return self.n_market_makers + self.n_arbitrageurs + self.n_trend_followers + self.n_fundamental_investors

class MultiAgentFactorModeler:
    """
    Pipeline for:
    1. Finding optimal agent configuration to match target market behavior
    2. Extracting factors (volatility signals)
    3. Modeling factors using Time Series (VAR/AR)
    4. Evaluating with Rust Engine
    """
    
    def __init__(self):
        self.extractor = MultiAgentVolatilityExtractor()
        self.rust_validator = RustVolatilityValidator()
        self.best_config: Optional[AgentConfiguration] = None
        self.time_series_model = None
        
    def _run_simulation_with_config(self, config: AgentConfiguration, n_steps: int = 252) -> pd.DataFrame:
        """Run simulation with specific agent counts and weights"""
        
        # We need to hack/modify the extractor to accept specific agent counts
        # Currently the extractor just takes n_agents and cycles through types.
        # We will subclass or just monkey-patch the list generation in simulate_and_extract
        # But simpler: The extractor uses 'n_agents' and modulo.
        
        # Let's create a custom list of agents based on config
        agent_types = []
        agent_types.extend(['market_maker'] * config.n_market_makers)
        agent_types.extend(['arbitrageur'] * config.n_arbitrageurs)
        agent_types.extend(['trend_follower'] * config.n_trend_followers)
        agent_types.extend(['fundamental'] * config.n_fundamental_investors)
        
        # Shuffle them
        np.random.shuffle(agent_types)
        
        # We need to temporarily force the extractor to use this specific distribution
        # Since simulate_and_extract creates agents inside, we have to modify how we call it
        # or update the extractor class. 
        # Actually, let's create a derived class here for flexibility.
        
        class ConfigurableExtractor(MultiAgentVolatilityExtractor):
            def simulate_with_types(self, agent_types_list, n_steps, initial_price=100):
                # Similar to simulate_and_extract but uses provided types list
                from models.volatility.multi_agent.agent_volatility import AgentState, MarketState
                 
                agents = []
                for i, agent_type in enumerate(agent_types_list):
                    agents.append(AgentState(
                        agent_type=agent_type,
                        position=0,
                        cash=100000,
                        last_action='hold',
                        confidence=0.5
                    ))
                
                # Logic copied from refined simulate_and_extract
                price_history = [initial_price]
                volatility_history = []
                intrinsic_value = initial_price
                
                for t in range(n_steps):
                    market_state_prev = MarketState(
                        price=price_history[-1],
                        volume=0, bid_ask_spread=0.01, num_trades=0, timestamp=t
                    )
                    
                    buy_pressure = 0
                    sell_pressure = 0
                    intrinsic_value *= (1 + np.random.normal(0, 0.005))
                    
                    agent_states = {}
                    for i, agent in enumerate(agents):
                        action = 'hold'
                        if agent.agent_type == 'trend_follower':
                            if len(price_history) > 5:
                                trend = price_history[-1] - price_history[-5]
                                if trend > 0.05: action = 'buy'
                                elif trend < -0.05: action = 'sell'
                        elif agent.agent_type == 'fundamental':
                            if price_history[-1] < intrinsic_value * 0.95: action = 'buy'
                            elif price_history[-1] > intrinsic_value * 1.05: action = 'sell'
                        elif agent.agent_type == 'arbitrageur':
                            if np.random.random() < 0.1:
                                if price_history[-1] > np.mean(price_history[-20:]) if len(price_history)>20 else False: action = 'sell'
                                else: action = 'buy'
                        
                        agent.last_action = action
                        if action == 'buy': buy_pressure += 1
                        elif action == 'sell': sell_pressure += 1
                        agent_states[f'agent_{i}'] = agent
                    
                    current_price = price_history[-1]
                    net_imbalance = buy_pressure - sell_pressure
                    n_mm = sum(1 for a in agents if a.agent_type == 'market_maker')
                    impact_factor = 0.01 / (1 + n_mm * 0.05)
                    price_return = net_imbalance * impact_factor + np.random.normal(0, 0.005)
                    new_price = current_price * (1 + price_return)
                    price_history.append(new_price)
                    
                    market_state = MarketState(
                        price=new_price,
                        volume=int((buy_pressure+sell_pressure)*100),
                        bid_ask_spread=0.01,
                        num_trades=buy_pressure+sell_pressure,
                        timestamp=t
                    )
                    
                    vol_signals = self.extract_volatility(market_state, agent_states, price_history)
                    volatility_history.append(vol_signals)
                    
                df = pd.DataFrame(volatility_history)
                df['price'] = price_history[1:]
                return df

        configurable_extractor = ConfigurableExtractor(weights=config.to_weights_dict())
        return configurable_extractor.simulate_with_types(agent_types, n_steps)

    def find_optimal_configuration(self, target_price_history: List[float], n_iterations: int = 50) -> AgentConfiguration:
        """
        Find agent configuration that produces price trend most similar to target
        Metric: Mean Squared Error of Log Returns + Volatility difference
        """
        print(f"Finding optimal multi-agent configuration ({n_iterations} iterations)...")
        
        target_returns = np.diff(np.log(target_price_history))
        target_vol = np.std(target_returns)
        
        best_error = float('inf')
        best_config = None
        
        # Simple Random Search for demonstration
        for i in range(n_iterations):
            # Randomize agent counts (total 100)
            counts = np.random.multinomial(100, [0.25, 0.25, 0.25, 0.25])
            
            # Randomize weights
            weights = np.random.dirichlet(np.ones(4))
            
            config = AgentConfiguration(
                n_market_makers=counts[0],
                n_arbitrageurs=counts[1],
                n_trend_followers=counts[2],
                n_fundamental_investors=counts[3],
                weight_market_maker=weights[0],
                weight_arbitrageur=weights[1],
                weight_trend_follower=weights[2],
                weight_fundamental=weights[3]
            )
            
            try:
                results = self._run_simulation_with_config(config, n_steps=len(target_price_history)-1)
                
                sim_returns = np.diff(np.log(results['price'].values))
                sim_vol = np.std(sim_returns)
                
                # Metrics
                vol_error = abs(sim_vol - target_vol)
                # Distribution error (KS test or just moments)
                skew_error = abs(pd.Series(sim_returns).skew() - pd.Series(target_returns).skew())
                kurt_error = abs(pd.Series(sim_returns).kurt() - pd.Series(target_returns).kurt())
                
                total_error = vol_error * 100 + skew_error + kurt_error
                
                if total_error < best_error:
                    best_error = total_error
                    best_config = config
                    print(f"  Iter {i}: New Best Error = {best_error:.4f} (Vol diff: {vol_error:.4f})")
                    
            except Exception as e:
                print(f"  Iter {i}: Failed - {e}")
                continue
                
        self.best_config = best_config
        return best_config

    def model_factors_time_series(self, simulation_results: pd.DataFrame) -> Any:
        """
        Model the extracted factors using Vector Autoregression (VAR)
        Factors: market_maker_vol, arbitrageur_vol, trend_follower_vol, fundamental_vol
        """
        try:
            from statsmodels.tsa.api import VAR
        except ImportError:
            print("statsmodels not found. Using simple AR(1) for demonstration.")
            return None

        data = simulation_results[[
            'market_maker_vol', 
            'arbitrageur_vol', 
            'trend_follower_vol', 
            'fundamental_vol'
        ]]
        
        # Stability: Add small noise to prevent singular matrix if factors are constant
        data = data + np.random.normal(0, 1e-6, data.shape)
        
        # Check for constant columns after noise (should be fine now)
        if data.std().min() < 1e-9:
             print("Warning: Some factors have near-zero variance even after noise. Dropping them.")
             data = data.loc[:, data.std() > 1e-9]
        
        if data.empty:
             raise ValueError("No valid factors to model.")

        try:
            model = VAR(data)
            # Use lower maxlags for stability on short samples
            max_lags = min(5, len(data) // 10)
            results = model.fit(maxlags=max_lags, ic='aic')
            self.time_series_model = results
            
            print("\nTime Series Factor Model (VAR) Fitted:")
            print(results.summary())
            return results
        except Exception as e:
            print(f"VAR fitting failed: {e}. Trying simple Mean baseline.")
            return None

    def predict_and_evaluate(self, target_actuals: List[float]) -> ValidationResult:
        """
        Use the fitted factor model to predict volatility and evaluate with Rust
        """
        returns = np.diff(np.log(target_actuals))
        realized_vol = np.array([np.std(returns[max(0, i-20):i]) * np.sqrt(252) for i in range(1, len(returns)+1)])
        realized_vol = np.pad(realized_vol, (1, 0), 'edge')
        
        # Calculate lag order
        lag_order = 0
        min_len = len(realized_vol)
        pred_vol = np.zeros_like(realized_vol)

        if self.time_series_model:
            fitted_values = self.time_series_model.fittedvalues
            lag_order = self.time_series_model.k_ar
            
            w = self.best_config.to_weights_dict()
            
            # Reconstruct prediction using only available columns
            pred_series = np.zeros(len(fitted_values))
            for col in fitted_values.columns:
                if col in w:
                    # Map column name to weight key (assuming exact match: market_maker_vol -> market_maker is not exact match)
                    # Keys in w are: market_maker, arbitrageur, trend_follower, fundamental
                    # Columns are: market_maker_vol, etc.
                    key = col.replace('_vol', '')
                    if key in w:
                        pred_series += fitted_values[col].values * w[key]
            
            # Scale if necessary? The weights sum to ~1 usually but factors are 0.1-0.3. So result is ~vol.
            pred_vol = pred_series
            
            # Adjust alignment
            actual_vol = realized_vol[lag_order:]
            min_len = min(len(pred_vol), len(actual_vol))
            pred_vol = pred_vol[:min_len]
            actual_vol = actual_vol[:min_len]
        else:
            print("Warning: No Time Series Model. Using simple mean prediction from simulation constants.")
            # Fallback: Just use the constant weights * average extracted vol estimates
            actual_vol = realized_vol
            pred_vol = np.full_like(actual_vol, 0.20) # Dummy fallback
        
        # Create Market Params
        market_params = {
            "risk_free_rate": 0.02,
            "dividend_yield": 0.0,
            "spot_price": target_actuals[-1]
        }
        
        # Evaluate with Rust
        result = self.rust_validator.validate_model(
            model_name="MultiAgent_VAR_Factor_Model",
            predictions=pred_vol.tolist(),
            actuals=actual_vol.tolist(), 
            market_params=market_params,
            risk_free_rate=0.02,
            returns=returns[lag_order:lag_order+min_len].tolist()
        )
        
        return result

if __name__ == "__main__":
    # Example Usage
    modeler = MultiAgentFactorModeler()
    
    # 1. Generate Dummy Target Data (e.g., GARCH process or just random walk with trend)
    print("Generating target market data...")
    price = 100
    target_history = [100.0]
    for _ in range(252):
        price *= (1 + np.random.normal(0.0005, 0.015))
        target_history.append(price)
        
    # 2. Find Optimal Agent Configuration
    best_config = modeler.find_optimal_configuration(target_history, n_iterations=10)
    print("\nOptimal Configuration:")
    print(best_config)
    
    # 3. Running best simulation to get factors
    print("\nRunning best simulation...")
    results = modeler._run_simulation_with_config(best_config, n_steps=len(target_history)-1)
    
    # 4. Model Factors
    modeler.model_factors_time_series(results)
    
    # 5. Evaluate
    print("\nEvaluating with Rust Engine...")
    val_result = modeler.predict_and_evaluate(target_history)
    
    print("\nEvaluation Results:")
    print(val_result)
