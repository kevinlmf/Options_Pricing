"""
Agent Interaction Engine

Coordinates multi-agent interactions and price formation to generate
realistic option pricing deviations. This is the core engine that
translates agent behaviors into market outcomes and risk metrics.

Key Features:
- Market equilibrium computation with friction effects
- Volatility smile/skew generation through agent imbalances
- Risk regime identification (stable vs unstable equilibria)
- Stress testing capabilities for quantitative risk management
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import warnings

from .base_agent import BaseAgent, AgentType, MarketState
from .market_maker import MarketMaker, MarketMakerParameters
from .arbitrageur import Arbitrageur, ArbitrageParameters
from .noise_trader import NoiseTrader, NoiseTraderParameters


class MarketRegime(Enum):
    """Market stability regimes"""
    STABLE = "stable"           # Deviations quickly corrected
    UNSTABLE = "unstable"       # Persistent large deviations
    STRESSED = "stressed"       # Extreme deviations, limited arbitrage
    ILLIQUID = "illiquid"       # Wide spreads, little trading


@dataclass
class MarketEquilibrium:
    """Results of market equilibrium computation"""

    # Price formation results
    equilibrium_prices: Dict[Tuple[float, float], float]
    pricing_deviations: Dict[Tuple[float, float], float]
    implied_volatilities: Dict[Tuple[float, float], float]
    bid_ask_spreads: Dict[Tuple[float, float], Tuple[float, float]]

    # Market structure metrics
    market_regime: MarketRegime
    liquidity_score: float  # 0-1, higher = more liquid
    arbitrage_capacity: float  # Available arbitrage capital
    deviation_persistence: float  # How long deviations last

    # Risk metrics for quantitative applications
    smile_skew_parameters: Dict[str, float]  # Fitted smile parameters
    tail_risk_multiplier: float  # How much model underestimates tail risk
    hedge_effectiveness: float  # Expected hedge ratio deterioration
    systemic_risk_indicator: float  # 0-1, higher = more systemic risk

    # Agent summary statistics
    agent_states: Dict[str, Dict[str, Any]]
    market_concentration: Dict[AgentType, float]  # Share of market activity


class AgentInteractionEngine:
    """
    Multi-Agent Market Simulation Engine

    Orchestrates interactions between different agent types to produce
    realistic option pricing deviations and market microstructure effects.

    Designed specifically for quantitative risk management applications:
    - Stress testing under different agent configurations
    - Identifying market fragility conditions
    - Calibrating risk models with realistic frictions
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize interaction engine.

        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration including agent parameters and market settings
        """
        self.config = config

        # Initialize agents
        self.agents: List[BaseAgent] = []
        self.market_makers: List[MarketMaker] = []
        self.arbitrageurs: List[Arbitrageur] = []
        self.noise_traders: List[NoiseTrader] = []

        self._initialize_agents()

        # Market state tracking
        self.current_market_state: Optional[MarketState] = None
        self.equilibrium_history: List[MarketEquilibrium] = []

        # Risk metrics tracking for quantitative applications
        self.deviation_statistics: Dict[str, List[float]] = {}
        self.regime_transitions: List[Tuple[float, MarketRegime, MarketRegime]] = []
        self.stress_test_results: Dict[str, Dict] = {}

    def _initialize_agents(self) -> None:
        """Initialize agents based on configuration."""

        # Create market makers
        mm_config = self.config.get('market_makers', {})
        for i in range(mm_config.get('count', 2)):
            params = MarketMakerParameters(
                max_inventory=mm_config.get('max_inventory', 100.0),
                inventory_risk_aversion=mm_config.get('risk_aversion', [0.01, 0.02])[i % 2],
                base_spread=mm_config.get('base_spread', 0.02),
                volatility_adjustment=mm_config.get('volatility_adjustment', 0.5)
            )

            agent_id = f"MM_{i}"
            mm = MarketMaker(agent_id, params, mm_config.get('initial_cash', 2000000.0))

            self.agents.append(mm)
            self.market_makers.append(mm)

        # Create arbitrageurs
        arb_config = self.config.get('arbitrageurs', {})
        for i in range(arb_config.get('count', 1)):
            params = ArbitrageParameters(
                min_deviation_threshold=arb_config.get('deviation_threshold', 0.01),
                max_capital_per_trade=arb_config.get('max_capital_per_trade', 100000.0),
                transaction_cost_rate=arb_config.get('transaction_costs', 0.002)
            )

            agent_id = f"ARB_{i}"
            arb = Arbitrageur(agent_id, params, arb_config.get('initial_cash', 5000000.0))

            self.agents.append(arb)
            self.arbitrageurs.append(arb)

        # Create noise traders
        noise_config = self.config.get('noise_traders', {})
        behavior_types = noise_config.get('behavior_types', ['momentum', 'mean_reversion', 'herding'])

        for i in range(noise_config.get('count', 10)):
            from .noise_trader import NoiseTraderBehavior
            behavior = NoiseTraderBehavior(behavior_types[i % len(behavior_types)])

            params = NoiseTraderParameters(
                behavior_type=behavior,
                base_trade_frequency=noise_config.get('trade_frequency', 0.1),
                trade_size_mean=noise_config.get('trade_size', 10.0),
                overconfidence_factor=noise_config.get('overconfidence', 1.5)
            )

            agent_id = f"NOISE_{i}"
            noise = NoiseTrader(agent_id, params, noise_config.get('initial_cash', 500000.0))

            self.agents.append(noise)
            self.noise_traders.append(noise)

    def simulate_market_period(self, market_state: MarketState) -> MarketEquilibrium:
        """
        Simulate one market period with agent interactions.

        Parameters:
        -----------
        market_state : MarketState
            Current market conditions (prices, volatility, etc.)

        Returns:
        --------
        MarketEquilibrium
            Resulting market equilibrium with pricing deviations
        """
        self.current_market_state = market_state

        # Step 1: All agents observe market
        for agent in self.agents:
            agent.observe_market(market_state)

        # Step 2: Collect agent decisions
        agent_decisions = {}
        for agent in self.agents:
            decision = agent.make_decision(market_state)
            agent_decisions[agent.state.agent_id] = decision

        # Step 3: Process market interactions and compute equilibrium
        equilibrium = self._compute_market_equilibrium(market_state, agent_decisions)

        # Step 4: Execute trades and update agent positions
        self._execute_market_transactions(equilibrium, agent_decisions)

        # Step 5: Update performance tracking
        for agent in self.agents:
            agent.record_performance(market_state)

        # Step 6: Record equilibrium for analysis
        self.equilibrium_history.append(equilibrium)
        self._update_statistics(equilibrium)

        return equilibrium

    def _compute_market_equilibrium(self, market_state: MarketState,
                                  agent_decisions: Dict[str, Dict]) -> MarketEquilibrium:
        """
        Compute market equilibrium given agent decisions.

        This is where the magic happens - agent behaviors translate into pricing deviations.
        """

        # Initialize with theoretical prices
        equilibrium_prices = market_state.theoretical_prices.copy()
        bid_ask_spreads = {}
        order_flow = {}

        # Process market maker quotes
        mm_quotes = self._aggregate_market_maker_quotes(agent_decisions)

        # Process demand/supply from other agents
        net_demand = self._calculate_net_demand(agent_decisions, market_state)

        # Compute final prices incorporating all effects
        pricing_deviations = {}
        implied_volatilities = {}

        for (strike, expiry), theoretical_price in market_state.theoretical_prices.items():

            # Get market maker bid-ask
            mm_bid, mm_ask = mm_quotes.get((strike, expiry), (theoretical_price, theoretical_price))

            # Apply demand/supply pressure
            net_flow = net_demand.get((strike, expiry), 0.0)

            # Price impact from order flow (linear impact model)
            impact = self._calculate_price_impact(net_flow, theoretical_price)

            # Final equilibrium price
            if net_flow > 0:  # Net buying pressure -> price moves toward ask + impact
                eq_price = mm_ask + impact
            else:  # Net selling pressure -> price moves toward bid + impact
                eq_price = mm_bid + impact

            equilibrium_prices[(strike, expiry)] = max(0.01, eq_price)  # Floor at $0.01
            pricing_deviations[(strike, expiry)] = eq_price - theoretical_price
            bid_ask_spreads[(strike, expiry)] = (mm_bid, mm_ask)
            order_flow[(strike, expiry)] = net_flow

            # Compute implied volatility from equilibrium price
            implied_vol = self._calculate_implied_volatility(
                eq_price, strike, expiry, market_state.underlying_price, market_state.risk_free_rate
            )
            implied_volatilities[(strike, expiry)] = implied_vol

        # Assess market regime and risk metrics
        market_regime = self._assess_market_regime(pricing_deviations, bid_ask_spreads)
        risk_metrics = self._calculate_risk_metrics(pricing_deviations, implied_volatilities, market_state)

        # Build equilibrium result
        equilibrium = MarketEquilibrium(
            equilibrium_prices=equilibrium_prices,
            pricing_deviations=pricing_deviations,
            implied_volatilities=implied_volatilities,
            bid_ask_spreads=bid_ask_spreads,
            market_regime=market_regime,
            **risk_metrics
        )

        return equilibrium

    def _aggregate_market_maker_quotes(self, agent_decisions: Dict[str, Dict]) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """Aggregate market maker quotes into best bid/offer."""

        all_quotes = {}

        # Collect all market maker quotes
        for agent_id, decision in agent_decisions.items():
            if decision.get('action') == 'quote':
                quotes = decision.get('quotes', {})
                for key, (bid, ask) in quotes.items():
                    if key not in all_quotes:
                        all_quotes[key] = {'bids': [], 'asks': []}
                    all_quotes[key]['bids'].append(bid)
                    all_quotes[key]['asks'].append(ask)

        # Compute best bid/offer for each option
        best_quotes = {}
        for key, quote_lists in all_quotes.items():
            best_bid = max(quote_lists['bids']) if quote_lists['bids'] else 0.01
            best_ask = min(quote_lists['asks']) if quote_lists['asks'] else 1.0
            best_quotes[key] = (best_bid, best_ask)

        return best_quotes

    def _calculate_net_demand(self, agent_decisions: Dict[str, Dict], market_state: MarketState) -> Dict[Tuple[float, float], float]:
        """Calculate net demand for each option from non-market-maker agents."""

        net_demand = {}

        for agent_id, decision in agent_decisions.items():
            action = decision.get('action')

            if action in ['buy', 'sell', 'arbitrage']:
                instrument = decision.get('instrument', '')
                quantity = decision.get('quantity', 0.0)

                # Parse instrument to get strike and expiry
                key = self._parse_instrument(instrument)
                if key is None:
                    continue

                # Add to net demand (positive = buying pressure, negative = selling pressure)
                demand = quantity if action in ['buy', 'arbitrage'] else -quantity

                if decision.get('direction') == 'sell':  # For arbitrageur decisions
                    demand = -demand

                net_demand[key] = net_demand.get(key, 0.0) + demand

        return net_demand

    def _parse_instrument(self, instrument: str) -> Optional[Tuple[float, float]]:
        """Parse instrument string to (strike, expiry) tuple."""
        try:
            parts = instrument.split('_')
            if len(parts) >= 3 and parts[0] == 'CALL':
                strike = float(parts[1])
                expiry = float(parts[2])
                return (strike, expiry)
        except:
            pass
        return None

    def _calculate_price_impact(self, order_flow: float, theoretical_price: float) -> float:
        """Calculate price impact from order flow."""

        # Linear impact model: larger trades have proportionally larger impact
        impact_coefficient = self.config.get('market_structure', {}).get('impact_coefficient', 0.001)

        # Impact scales with square root of flow (realistic market impact)
        impact = impact_coefficient * theoretical_price * np.sign(order_flow) * np.sqrt(abs(order_flow))

        return impact

    def _calculate_implied_volatility(self, option_price: float, strike: float, expiry: float,
                                    spot: float, rate: float) -> float:
        """Calculate implied volatility using simplified Black-Scholes inversion."""

        # Simplified IV calculation - in practice would use Newton-Raphson
        from scipy.optimize import minimize_scalar

        def bs_price_error(vol):
            try:
                # Import Black-Scholes calculator
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                from models.black_scholes import BlackScholesModel, BSParameters

                params = BSParameters(S0=spot, K=strike, T=expiry, r=rate, sigma=vol)
                model = BlackScholesModel(params)
                theoretical = model.call_price()
                return abs(theoretical - option_price)
            except:
                return 1000.0  # Large error if calculation fails

        try:
            result = minimize_scalar(bs_price_error, bounds=(0.01, 2.0), method='bounded')
            return max(0.01, min(2.0, result.x))  # Bound between 1% and 200%
        except:
            return 0.2  # Default to 20% if calculation fails

    def _assess_market_regime(self, deviations: Dict[Tuple[float, float], float],
                            spreads: Dict[Tuple[float, float], Tuple[float, float]]) -> MarketRegime:
        """Assess current market stability regime."""

        if not deviations:
            return MarketRegime.STABLE

        # Calculate regime metrics
        avg_deviation = np.mean([abs(d) for d in deviations.values()])
        max_deviation = max([abs(d) for d in deviations.values()])

        avg_spread = np.mean([(ask - bid) / ((ask + bid) / 2) for bid, ask in spreads.values() if ask > bid])

        # Classify regime based on thresholds
        if max_deviation > 0.1 or avg_spread > 0.1:  # 10% deviations or spreads
            return MarketRegime.STRESSED
        elif avg_deviation > 0.03 or avg_spread > 0.05:  # 3% avg deviation or 5% spreads
            return MarketRegime.UNSTABLE
        elif avg_spread > 0.03:  # Wide spreads but small deviations
            return MarketRegime.ILLIQUID
        else:
            return MarketRegime.STABLE

    def _calculate_risk_metrics(self, deviations: Dict, implied_vols: Dict, market_state: MarketState) -> Dict:
        """Calculate quantitative risk metrics for practical applications."""

        if not deviations:
            return {
                'liquidity_score': 1.0,
                'arbitrage_capacity': 1.0,
                'deviation_persistence': 0.0,
                'smile_skew_parameters': {},
                'tail_risk_multiplier': 1.0,
                'hedge_effectiveness': 1.0,
                'systemic_risk_indicator': 0.0,
                'agent_states': {},
                'market_concentration': {}
            }

        # Liquidity score (0-1, higher = more liquid)
        avg_deviation = np.mean([abs(d) for d in deviations.values()])
        liquidity_score = max(0.0, 1.0 - avg_deviation / 0.05)  # Perfect at 0% deviation, zero at 5%

        # Arbitrage capacity (fraction of deviations that can be arbitraged)
        total_arbitrage_capital = sum(arb.state.cash_position for arb in self.arbitrageurs)
        market_value = sum(market_state.option_prices.values()) if market_state.option_prices else 1.0
        arbitrage_capacity = min(1.0, total_arbitrage_capital / market_value)

        # Deviation persistence (how long deviations typically last)
        if len(self.equilibrium_history) > 5:
            recent_deviations = [np.mean([abs(d) for d in eq.pricing_deviations.values()])
                               for eq in self.equilibrium_history[-5:]]
            deviation_persistence = np.std(recent_deviations)
        else:
            deviation_persistence = 0.0

        # Volatility smile/skew parameters
        smile_skew = self._fit_smile_parameters(implied_vols, market_state)

        # Tail risk multiplier (how much standard models underestimate risk)
        tail_risk_multiplier = 1.0 + avg_deviation * 2  # Heuristic: deviations indicate model risk

        # Hedge effectiveness (expected deterioration due to frictions)
        hedge_effectiveness = max(0.5, 1.0 - avg_deviation)

        # Systemic risk indicator
        regime_risk = {'stable': 0.1, 'unstable': 0.5, 'stressed': 0.9, 'illiquid': 0.7}
        market_regime = self._assess_market_regime(deviations, {})
        systemic_risk = regime_risk.get(market_regime.value, 0.5)

        # Agent states summary
        agent_states = {agent.state.agent_id: {
            'type': agent.state.agent_type.value,
            'pnl': agent.state.total_pnl,
            'positions': len([p for p in agent.state.inventory.values() if abs(p) > 0.01])
        } for agent in self.agents}

        # Market concentration by agent type
        total_volume = sum(len(agent.state.inventory) for agent in self.agents) or 1
        market_concentration = {
            agent_type: sum(len(agent.state.inventory) for agent in self.agents
                          if agent.state.agent_type == agent_type) / total_volume
            for agent_type in AgentType
        }

        return {
            'liquidity_score': liquidity_score,
            'arbitrage_capacity': arbitrage_capacity,
            'deviation_persistence': deviation_persistence,
            'smile_skew_parameters': smile_skew,
            'tail_risk_multiplier': tail_risk_multiplier,
            'hedge_effectiveness': hedge_effectiveness,
            'systemic_risk_indicator': systemic_risk,
            'agent_states': agent_states,
            'market_concentration': market_concentration
        }

    def _fit_smile_parameters(self, implied_vols: Dict, market_state: MarketState) -> Dict[str, float]:
        """Fit volatility smile parameters for risk modeling."""

        if not implied_vols:
            return {}

        # Group by expiry and fit smile for each
        smile_params = {}

        expiries = set([key[1] for key in implied_vols.keys()])

        for expiry in expiries:
            # Get strikes and IVs for this expiry
            strikes_ivs = [(key[0], iv) for key, iv in implied_vols.items() if key[1] == expiry]

            if len(strikes_ivs) < 3:  # Need at least 3 points
                continue

            strikes, ivs = zip(*strikes_ivs)
            strikes = np.array(strikes)
            ivs = np.array(ivs)

            # Normalize strikes by spot (moneyness)
            moneyness = np.log(strikes / market_state.underlying_price)

            try:
                # Fit quadratic smile: IV(m) = a + b*m + c*m^2
                coeffs = np.polyfit(moneyness, ivs, 2)

                atm_vol = coeffs[2]  # ATM volatility (constant term after centering)
                skew = coeffs[1]     # First derivative (skew)
                convexity = coeffs[0] * 2  # Second derivative (smile convexity)

                smile_params[f'expiry_{expiry:.2f}'] = {
                    'atm_vol': atm_vol,
                    'skew': skew,
                    'convexity': convexity
                }

            except:
                continue  # Skip if fitting fails

        return smile_params

    def _execute_market_transactions(self, equilibrium: MarketEquilibrium, agent_decisions: Dict) -> None:
        """Execute trades based on equilibrium and update agent positions."""

        # For now, simplified execution - assume all desired trades execute at equilibrium prices
        for agent_id, decision in agent_decisions.items():
            agent = next((a for a in self.agents if a.state.agent_id == agent_id), None)
            if agent is None:
                continue

            action = decision.get('action')
            if action not in ['buy', 'sell', 'arbitrage']:
                continue

            # Get trade details
            instrument = decision.get('instrument', '')
            quantity = decision.get('quantity', 0.0)

            key = self._parse_instrument(instrument)
            if key is None or quantity <= 0:
                continue

            # Get execution price
            execution_price = equilibrium.equilibrium_prices.get(key)
            if execution_price is None:
                continue

            # Execute trade
            trade_quantity = quantity if action in ['buy'] else -quantity
            if decision.get('direction') == 'sell':  # For arbitrageur
                trade_quantity = -trade_quantity

            agent.update_position(instrument, trade_quantity, execution_price)

    def _update_statistics(self, equilibrium: MarketEquilibrium) -> None:
        """Update running statistics for analysis."""

        # Track deviations
        if 'avg_deviation' not in self.deviation_statistics:
            self.deviation_statistics['avg_deviation'] = []

        avg_dev = np.mean([abs(d) for d in equilibrium.pricing_deviations.values()])
        self.deviation_statistics['avg_deviation'].append(avg_dev)

        # Track regime transitions
        if len(self.equilibrium_history) > 1:
            prev_regime = self.equilibrium_history[-2].market_regime
            curr_regime = equilibrium.market_regime

            if prev_regime != curr_regime:
                self.regime_transitions.append((len(self.equilibrium_history), prev_regime, curr_regime))

    def run_stress_test(self, stress_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run stress test by modifying agent configurations and measuring outcomes.

        Critical for quantitative risk management.
        """

        # Save original configuration
        original_config = self.config.copy()

        # Apply stress scenario
        if 'reduce_arbitrage_capital' in stress_scenario:
            factor = stress_scenario['reduce_arbitrage_capital']
            for arb in self.arbitrageurs:
                arb.state.cash_position *= factor

        if 'increase_noise_trader_activity' in stress_scenario:
            factor = stress_scenario['increase_noise_trader_activity']
            for noise in self.noise_traders:
                noise.params.base_trade_frequency *= factor

        if 'widen_mm_spreads' in stress_scenario:
            factor = stress_scenario['widen_mm_spreads']
            for mm in self.market_makers:
                mm.params.base_spread *= factor

        # Run simulation and record results
        stress_results = {
            'scenario': stress_scenario,
            'equilibria': [],
            'max_deviation': 0.0,
            'avg_deviation': 0.0,
            'regime_distribution': {},
            'systemic_risk_peak': 0.0
        }

        # This would be called by external simulation loop
        # Results would be populated during actual stress simulation

        # Restore original configuration
        self.config = original_config

        return stress_results

    def get_market_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive market analysis for quantitative applications.
        """

        if not self.equilibrium_history:
            return {}

        recent_equilibria = self.equilibrium_history[-100:]  # Last 100 periods

        # Deviation analysis
        all_deviations = []
        for eq in recent_equilibria:
            all_deviations.extend([abs(d) for d in eq.pricing_deviations.values()])

        deviation_stats = {
            'mean': np.mean(all_deviations) if all_deviations else 0.0,
            'std': np.std(all_deviations) if all_deviations else 0.0,
            'max': max(all_deviations) if all_deviations else 0.0,
            'percentile_95': np.percentile(all_deviations, 95) if all_deviations else 0.0
        }

        # Regime analysis
        regime_counts = {}
        for eq in recent_equilibria:
            regime = eq.market_regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        regime_distribution = {k: v / len(recent_equilibria) for k, v in regime_counts.items()}

        # Risk metrics evolution
        risk_evolution = {
            'liquidity_scores': [eq.liquidity_score for eq in recent_equilibria],
            'systemic_risk': [eq.systemic_risk_indicator for eq in recent_equilibria],
            'tail_risk_multipliers': [eq.tail_risk_multiplier for eq in recent_equilibria]
        }

        return {
            'deviation_statistics': deviation_stats,
            'regime_distribution': regime_distribution,
            'risk_evolution': risk_evolution,
            'regime_transitions': len(self.regime_transitions),
            'current_regime': recent_equilibria[-1].market_regime.value if recent_equilibria else 'unknown'
        }