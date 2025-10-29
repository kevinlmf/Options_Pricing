"""
Option Risk Management

This module provides comprehensive risk analysis for option portfolios,
including Greeks-based risk measures and option-specific risk metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import norm
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.option_portfolio import OptionPortfolio, OptionPosition, OptionStrategy
from models.options_pricing.black_scholes import BlackScholesModel, BSParameters


class GreeksRiskAnalyzer:
    """
    Analyzer for Greeks-based risk metrics in option portfolios.
    """

    def __init__(self, portfolio: OptionPortfolio):
        """
        Initialize Greeks risk analyzer.

        Parameters:
        -----------
        portfolio : OptionPortfolio
            The option portfolio to analyze
        """
        self.portfolio = portfolio

    def delta_risk_analysis(self) -> Dict:
        """
        Analyze delta risk (price sensitivity).

        Returns:
        --------
        dict : Delta risk metrics
        """
        portfolio_delta = self.portfolio.portfolio_greeks['delta']
        positions = self.portfolio.all_positions

        # Calculate delta by underlying
        delta_by_symbol = {}
        for position in positions:
            symbol = position.symbol
            pos_delta = position.greeks['delta']

            if symbol not in delta_by_symbol:
                delta_by_symbol[symbol] = 0
            delta_by_symbol[symbol] += pos_delta

        # Delta risk metrics
        total_delta = abs(portfolio_delta)
        max_delta_exposure = max(abs(delta) for delta in delta_by_symbol.values()) if delta_by_symbol else 0

        return {
            'portfolio_delta': portfolio_delta,
            'total_delta_exposure': total_delta,
            'max_delta_exposure_by_symbol': max_delta_exposure,
            'delta_by_symbol': delta_by_symbol,
            'delta_neutrality': abs(portfolio_delta) < 0.1  # Consider neutral if |delta| < 0.1
        }

    def gamma_risk_analysis(self) -> Dict:
        """
        Analyze gamma risk (delta sensitivity).

        Returns:
        --------
        dict : Gamma risk metrics
        """
        portfolio_gamma = self.portfolio.portfolio_greeks['gamma']
        positions = self.portfolio.all_positions

        # Calculate gamma by underlying
        gamma_by_symbol = {}
        for position in positions:
            symbol = position.symbol
            pos_gamma = position.greeks['gamma']

            if symbol not in gamma_by_symbol:
                gamma_by_symbol[symbol] = 0
            gamma_by_symbol[symbol] += pos_gamma

        # Gamma risk scenarios - how delta changes for 1% price moves
        delta_change_scenarios = {}
        for symbol, gamma in gamma_by_symbol.items():
            # Find a representative position for this symbol to get current price
            sample_pos = next((pos for pos in positions if pos.symbol == symbol), None)
            if sample_pos:
                spot_price = sample_pos.underlying_price
                price_change_1pct = spot_price * 0.01
                delta_change = gamma * price_change_1pct
                delta_change_scenarios[symbol] = {
                    'gamma': gamma,
                    'delta_change_1pct_up': delta_change,
                    'delta_change_1pct_down': -delta_change
                }

        return {
            'portfolio_gamma': portfolio_gamma,
            'gamma_by_symbol': gamma_by_symbol,
            'delta_change_scenarios': delta_change_scenarios,
            'high_gamma_risk': abs(portfolio_gamma) > 10  # Threshold for high gamma risk
        }

    def vega_risk_analysis(self) -> Dict:
        """
        Analyze vega risk (volatility sensitivity).

        Returns:
        --------
        dict : Vega risk metrics
        """
        portfolio_vega = self.portfolio.portfolio_greeks['vega']
        positions = self.portfolio.all_positions

        # Calculate vega by underlying and expiry
        vega_by_symbol = {}
        vega_by_expiry = {}

        for position in positions:
            symbol = position.symbol
            expiry = position.expiry
            pos_vega = position.greeks['vega']

            # By symbol
            if symbol not in vega_by_symbol:
                vega_by_symbol[symbol] = 0
            vega_by_symbol[symbol] += pos_vega

            # By expiry bucket
            expiry_bucket = self._get_expiry_bucket(expiry)
            if expiry_bucket not in vega_by_expiry:
                vega_by_expiry[expiry_bucket] = 0
            vega_by_expiry[expiry_bucket] += pos_vega

        # Vega risk scenarios - P&L change for volatility shifts
        volatility_scenarios = {
            'vol_up_1pct': portfolio_vega * 0.01,
            'vol_up_5pct': portfolio_vega * 0.05,
            'vol_down_1pct': portfolio_vega * -0.01,
            'vol_down_5pct': portfolio_vega * -0.05,
            'vol_shock_up_10pct': portfolio_vega * 0.10,
            'vol_shock_down_10pct': portfolio_vega * -0.10
        }

        return {
            'portfolio_vega': portfolio_vega,
            'vega_by_symbol': vega_by_symbol,
            'vega_by_expiry': vega_by_expiry,
            'volatility_scenarios': volatility_scenarios,
            'high_vega_risk': abs(portfolio_vega) > 100  # Threshold for high vega risk
        }

    def theta_risk_analysis(self) -> Dict:
        """
        Analyze theta risk (time decay).

        Returns:
        --------
        dict : Theta risk metrics
        """
        portfolio_theta = self.portfolio.portfolio_greeks['theta']
        positions = self.portfolio.all_positions

        # Calculate theta by expiry
        theta_by_expiry = {}
        daily_decay = {}

        for position in positions:
            expiry = position.expiry
            pos_theta = position.greeks['theta']

            expiry_bucket = self._get_expiry_bucket(expiry)
            if expiry_bucket not in theta_by_expiry:
                theta_by_expiry[expiry_bucket] = 0
                daily_decay[expiry_bucket] = 0

            theta_by_expiry[expiry_bucket] += pos_theta
            daily_decay[expiry_bucket] += pos_theta / 365  # Convert to daily decay

        # Time decay scenarios
        time_decay_scenarios = {
            'daily_theta_decay': portfolio_theta / 365,
            'weekly_theta_decay': portfolio_theta / 52,
            'monthly_theta_decay': portfolio_theta / 12
        }

        return {
            'portfolio_theta': portfolio_theta,
            'theta_by_expiry': theta_by_expiry,
            'time_decay_scenarios': time_decay_scenarios,
            'daily_decay_by_expiry': daily_decay,
            'high_theta_risk': abs(portfolio_theta) > 50  # Threshold for high theta risk
        }

    def rho_risk_analysis(self) -> Dict:
        """
        Analyze rho risk (interest rate sensitivity).

        Returns:
        --------
        dict : Rho risk metrics
        """
        portfolio_rho = self.portfolio.portfolio_greeks['rho']

        # Interest rate scenarios
        rate_scenarios = {
            'rate_up_25bp': portfolio_rho * 0.0025,
            'rate_up_50bp': portfolio_rho * 0.005,
            'rate_up_100bp': portfolio_rho * 0.01,
            'rate_down_25bp': portfolio_rho * -0.0025,
            'rate_down_50bp': portfolio_rho * -0.005,
            'rate_down_100bp': portfolio_rho * -0.01
        }

        return {
            'portfolio_rho': portfolio_rho,
            'rate_scenarios': rate_scenarios,
            'high_rho_risk': abs(portfolio_rho) > 100  # Threshold for high rho risk
        }

    def comprehensive_greeks_report(self) -> Dict:
        """
        Generate comprehensive Greeks risk report.

        Returns:
        --------
        dict : Complete Greeks analysis
        """
        return {
            'delta_analysis': self.delta_risk_analysis(),
            'gamma_analysis': self.gamma_risk_analysis(),
            'vega_analysis': self.vega_risk_analysis(),
            'theta_analysis': self.theta_risk_analysis(),
            'rho_analysis': self.rho_risk_analysis(),
            'portfolio_summary': self.portfolio.portfolio_greeks
        }

    def _get_expiry_bucket(self, expiry: float) -> str:
        """Categorize expiry into buckets."""
        if expiry <= 1/12:  # 1 month
            return "0-1M"
        elif expiry <= 3/12:  # 3 months
            return "1-3M"
        elif expiry <= 6/12:  # 6 months
            return "3-6M"
        elif expiry <= 1:  # 1 year
            return "6M-1Y"
        else:
            return "1Y+"


class OptionVaRCalculator:
    """
    Value-at-Risk calculator specifically designed for option portfolios.
    """

    def __init__(self, portfolio: OptionPortfolio):
        """
        Initialize option VaR calculator.

        Parameters:
        -----------
        portfolio : OptionPortfolio
            The option portfolio to analyze
        """
        self.portfolio = portfolio

    def delta_gamma_var(self,
                       confidence_level: float = 0.05,
                       time_horizon: int = 1,
                       underlying_volatilities: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate VaR using Delta-Gamma approximation.

        Parameters:
        -----------
        confidence_level : float
            Confidence level (e.g., 0.05 for 95% VaR)
        time_horizon : int
            Time horizon in days
        underlying_volatilities : dict, optional
            Volatilities for each underlying symbol

        Returns:
        --------
        dict : Delta-Gamma VaR results
        """
        positions = self.portfolio.all_positions
        if not positions:
            return {'var': 0, 'method': 'delta-gamma', 'confidence': 1-confidence_level}

        # Get portfolio Greeks
        greeks = self.portfolio.portfolio_greeks
        portfolio_delta = greeks['delta']
        portfolio_gamma = greeks['gamma']

        # Calculate VaR for each underlying separately, then combine
        var_components = {}
        total_var_squared = 0

        # Group positions by underlying
        positions_by_symbol = {}
        for position in positions:
            symbol = position.symbol
            if symbol not in positions_by_symbol:
                positions_by_symbol[symbol] = []
            positions_by_symbol[symbol].append(position)

        for symbol, symbol_positions in positions_by_symbol.items():
            # Calculate symbol-specific delta and gamma
            symbol_delta = sum(pos.greeks['delta'] for pos in symbol_positions)
            symbol_gamma = sum(pos.greeks['gamma'] for pos in symbol_positions)

            # Get volatility
            if underlying_volatilities and symbol in underlying_volatilities:
                vol = underlying_volatilities[symbol]
            else:
                # Use average volatility from positions
                vol = np.mean([pos.volatility for pos in symbol_positions])

            # Get current price
            current_price = symbol_positions[0].underlying_price

            # Scale volatility for time horizon
            vol_scaled = vol * np.sqrt(time_horizon / 252)

            # Delta-Gamma VaR calculation
            # P&L = delta * dS + 0.5 * gamma * dS^2
            # where dS ~ N(0, (S * vol_scaled)^2)

            alpha = norm.ppf(confidence_level)  # Critical value
            S_vol = current_price * vol_scaled

            # First order (delta) VaR
            delta_var = abs(symbol_delta * S_vol * alpha)

            # Second order (gamma) adjustment
            # For gamma term: E[gamma * dS^2] = gamma * sigma^2, Var[gamma * dS^2] ≈ 2*gamma^2*sigma^4
            gamma_adjustment = 0
            if abs(symbol_gamma) > 1e-10:
                gamma_adjustment = 0.5 * symbol_gamma * (S_vol ** 2)

            # Combined VaR for this symbol
            symbol_var = delta_var + gamma_adjustment
            var_components[symbol] = {
                'delta': symbol_delta,
                'gamma': symbol_gamma,
                'volatility': vol,
                'delta_var': delta_var,
                'gamma_adjustment': gamma_adjustment,
                'total_var': symbol_var
            }

            total_var_squared += symbol_var ** 2

        # Total portfolio VaR (assuming independence between underlyings)
        total_var = np.sqrt(total_var_squared)

        return {
            'var': total_var,
            'confidence': 1 - confidence_level,
            'time_horizon_days': time_horizon,
            'method': 'delta-gamma',
            'var_components': var_components,
            'portfolio_greeks': greeks
        }

    def monte_carlo_var(self,
                       confidence_level: float = 0.05,
                       time_horizon: int = 1,
                       n_simulations: int = 10000,
                       underlying_volatilities: Optional[Dict[str, float]] = None,
                       correlation_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate VaR using Monte Carlo simulation.

        Parameters:
        -----------
        confidence_level : float
            Confidence level
        time_horizon : int
            Time horizon in days
        n_simulations : int
            Number of Monte Carlo simulations
        underlying_volatilities : dict, optional
            Volatilities for each underlying
        correlation_matrix : np.ndarray, optional
            Correlation matrix between underlyings

        Returns:
        --------
        dict : Monte Carlo VaR results
        """
        positions = self.portfolio.all_positions
        if not positions:
            return {'var': 0, 'method': 'monte-carlo', 'confidence': 1-confidence_level}

        # Get unique symbols
        symbols = list(set(pos.symbol for pos in positions))
        n_assets = len(symbols)

        # Setup correlation matrix
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_assets)  # Assume independence

        # Get volatilities and current prices
        volatilities = []
        current_prices = {}

        for i, symbol in enumerate(symbols):
            symbol_positions = [pos for pos in positions if pos.symbol == symbol]
            current_prices[symbol] = symbol_positions[0].underlying_price

            if underlying_volatilities and symbol in underlying_volatilities:
                vol = underlying_volatilities[symbol]
            else:
                vol = np.mean([pos.volatility for pos in symbol_positions])
            volatilities.append(vol)

        volatilities = np.array(volatilities)

        # Scale for time horizon
        dt = time_horizon / 252
        vol_scaled = volatilities * np.sqrt(dt)

        # Generate correlated random shocks
        np.random.seed(42)  # For reproducibility
        random_shocks = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=correlation_matrix,
            size=n_simulations
        )

        # Calculate P&L for each simulation
        pnl_simulations = []

        for sim in range(n_simulations):
            total_pnl = 0

            for i, symbol in enumerate(symbols):
                # Price shock for this asset
                price_shock_pct = random_shocks[sim, i] * vol_scaled[i]
                price_shock = current_prices[symbol] * price_shock_pct

                # Calculate P&L from all positions in this symbol
                symbol_positions = [pos for pos in positions if pos.symbol == symbol]

                for position in symbol_positions:
                    # Update position with new underlying price
                    new_price = position.underlying_price + price_shock

                    # Create new BS model with updated price
                    new_params = BSParameters(
                        S0=new_price,
                        K=position.strike,
                        T=max(position.expiry - dt, 0.001),  # Reduce time to expiry
                        r=position.risk_free_rate,
                        sigma=position.volatility
                    )
                    new_model = BlackScholesModel(new_params)

                    # Calculate new option value
                    if position.option_type.value == 'call':
                        new_value = new_model.call_price()
                    else:
                        new_value = new_model.put_price()

                    # P&L for this position
                    old_value = position.current_value / position.quantity / position.position_type.value
                    position_pnl = (new_value - old_value) * position.quantity * position.position_type.value

                    total_pnl += position_pnl

            pnl_simulations.append(total_pnl)

        pnl_simulations = np.array(pnl_simulations)

        # Calculate VaR
        var = -np.percentile(pnl_simulations, confidence_level * 100)
        cvar = -np.mean(pnl_simulations[pnl_simulations <= -var])

        return {
            'var': var,
            'cvar': cvar,
            'confidence': 1 - confidence_level,
            'time_horizon_days': time_horizon,
            'method': 'monte-carlo',
            'n_simulations': n_simulations,
            'pnl_distribution': pnl_simulations.tolist(),
            'pnl_statistics': {
                'mean': np.mean(pnl_simulations),
                'std': np.std(pnl_simulations),
                'min': np.min(pnl_simulations),
                'max': np.max(pnl_simulations),
                'percentiles': {
                    '1%': np.percentile(pnl_simulations, 1),
                    '5%': np.percentile(pnl_simulations, 5),
                    '95%': np.percentile(pnl_simulations, 95),
                    '99%': np.percentile(pnl_simulations, 99)
                }
            }
        }


class OptionStressTester:
    """
    Stress testing framework for option portfolios.
    """

    def __init__(self, portfolio: OptionPortfolio):
        """
        Initialize option stress tester.

        Parameters:
        -----------
        portfolio : OptionPortfolio
            The option portfolio to test
        """
        self.portfolio = portfolio

    def market_crash_scenario(self, crash_magnitude: float = 0.20) -> Dict:
        """
        Simulate market crash scenario.

        Parameters:
        -----------
        crash_magnitude : float
            Magnitude of market crash (e.g., 0.20 for 20% drop)

        Returns:
        --------
        dict : Stress test results
        """
        positions = self.portfolio.all_positions
        original_value = self.portfolio.total_value

        crash_pnl = 0
        position_details = []

        for position in positions:
            # New underlying price after crash
            new_price = position.underlying_price * (1 - crash_magnitude)

            # Create new BS model
            new_params = BSParameters(
                S0=new_price,
                K=position.strike,
                T=position.expiry,
                r=position.risk_free_rate,
                sigma=position.volatility
            )
            new_model = BlackScholesModel(new_params)

            # Calculate new option value
            if position.option_type.value == 'call':
                new_option_value = new_model.call_price()
            else:
                new_option_value = new_model.put_price()

            new_position_value = new_option_value * position.quantity * position.position_type.value
            position_pnl = new_position_value - position.current_value

            crash_pnl += position_pnl

            position_details.append({
                'symbol': position.symbol,
                'option_type': position.option_type.value,
                'position_type': position.position_type.name,
                'original_value': position.current_value,
                'stressed_value': new_position_value,
                'pnl': position_pnl
            })

        return {
            'scenario': 'Market Crash',
            'crash_magnitude': crash_magnitude,
            'original_portfolio_value': original_value,
            'stressed_portfolio_value': original_value + crash_pnl,
            'total_pnl': crash_pnl,
            'pnl_percentage': (crash_pnl / original_value) * 100 if original_value != 0 else 0,
            'position_details': position_details
        }

    def volatility_shock_scenario(self, vol_shock: float = 0.10) -> Dict:
        """
        Simulate volatility shock scenario.

        Parameters:
        -----------
        vol_shock : float
            Volatility shock magnitude (e.g., 0.10 for +10% volatility)

        Returns:
        --------
        dict : Stress test results
        """
        positions = self.portfolio.all_positions
        original_value = self.portfolio.total_value

        vol_shock_pnl = 0
        position_details = []

        for position in positions:
            # New volatility after shock
            new_vol = position.volatility + vol_shock

            # Create new BS model
            new_params = BSParameters(
                S0=position.underlying_price,
                K=position.strike,
                T=position.expiry,
                r=position.risk_free_rate,
                sigma=new_vol
            )
            new_model = BlackScholesModel(new_params)

            # Calculate new option value
            if position.option_type.value == 'call':
                new_option_value = new_model.call_price()
            else:
                new_option_value = new_model.put_price()

            new_position_value = new_option_value * position.quantity * position.position_type.value
            position_pnl = new_position_value - position.current_value

            vol_shock_pnl += position_pnl

            position_details.append({
                'symbol': position.symbol,
                'option_type': position.option_type.value,
                'position_type': position.position_type.name,
                'original_value': position.current_value,
                'stressed_value': new_position_value,
                'pnl': position_pnl,
                'original_vol': position.volatility,
                'stressed_vol': new_vol
            })

        return {
            'scenario': 'Volatility Shock',
            'vol_shock': vol_shock,
            'original_portfolio_value': original_value,
            'stressed_portfolio_value': original_value + vol_shock_pnl,
            'total_pnl': vol_shock_pnl,
            'pnl_percentage': (vol_shock_pnl / original_value) * 100 if original_value != 0 else 0,
            'position_details': position_details
        }

    def time_decay_scenario(self, days_forward: int = 30) -> Dict:
        """
        Simulate time decay scenario.

        Parameters:
        -----------
        days_forward : int
            Number of days forward to simulate

        Returns:
        --------
        dict : Stress test results
        """
        positions = self.portfolio.all_positions
        original_value = self.portfolio.total_value

        time_decay_pnl = 0
        position_details = []

        for position in positions:
            # New time to expiry
            new_expiry = max(position.expiry - (days_forward / 365), 0.001)

            # Create new BS model
            new_params = BSParameters(
                S0=position.underlying_price,
                K=position.strike,
                T=new_expiry,
                r=position.risk_free_rate,
                sigma=position.volatility
            )
            new_model = BlackScholesModel(new_params)

            # Calculate new option value
            if position.option_type.value == 'call':
                new_option_value = new_model.call_price()
            else:
                new_option_value = new_model.put_price()

            new_position_value = new_option_value * position.quantity * position.position_type.value
            position_pnl = new_position_value - position.current_value

            time_decay_pnl += position_pnl

            position_details.append({
                'symbol': position.symbol,
                'option_type': position.option_type.value,
                'position_type': position.position_type.name,
                'original_value': position.current_value,
                'stressed_value': new_position_value,
                'pnl': position_pnl,
                'original_expiry': position.expiry,
                'stressed_expiry': new_expiry,
                'theta_impact': position_pnl / days_forward if days_forward > 0 else 0
            })

        return {
            'scenario': 'Time Decay',
            'days_forward': days_forward,
            'original_portfolio_value': original_value,
            'stressed_portfolio_value': original_value + time_decay_pnl,
            'total_pnl': time_decay_pnl,
            'pnl_percentage': (time_decay_pnl / original_value) * 100 if original_value != 0 else 0,
            'daily_theta_decay': time_decay_pnl / days_forward if days_forward > 0 else 0,
            'position_details': position_details
        }

    def comprehensive_stress_test(self) -> Dict:
        """
        Run comprehensive stress test with multiple scenarios.

        Returns:
        --------
        dict : Complete stress test results
        """
        scenarios = {}

        # Market crash scenarios
        for crash_size in [0.10, 0.20, 0.30]:
            scenarios[f'crash_{int(crash_size*100)}pct'] = self.market_crash_scenario(crash_size)

        # Volatility shock scenarios
        for vol_shock in [0.05, 0.10, 0.15, -0.05]:
            sign = "up" if vol_shock > 0 else "down"
            scenarios[f'vol_shock_{sign}_{int(abs(vol_shock)*100)}pct'] = self.volatility_shock_scenario(vol_shock)

        # Time decay scenarios
        for days in [7, 30, 60]:
            scenarios[f'time_decay_{days}d'] = self.time_decay_scenario(days)

        # Find worst case scenario
        worst_case = min(scenarios.items(), key=lambda x: x[1]['total_pnl'])
        best_case = max(scenarios.items(), key=lambda x: x[1]['total_pnl'])

        return {
            'scenarios': scenarios,
            'worst_case': {
                'scenario_name': worst_case[0],
                'results': worst_case[1]
            },
            'best_case': {
                'scenario_name': best_case[0],
                'results': best_case[1]
            },
            'original_portfolio_value': self.portfolio.total_value
        }