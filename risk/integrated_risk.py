"""
Integrated Risk Management

This module provides comprehensive risk management for portfolios containing
both stocks and options, integrating all risk factors into a unified framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.var_models import VaRModel
from risk.cvar_models import CVaRModel
from risk.portfolio_risk import PortfolioRiskMetrics
from risk.option_risk import GreeksRiskAnalyzer, OptionVaRCalculator, OptionStressTester
from models.option_portfolio import OptionPortfolio, OptionPosition, StrategyBuilder
from models.implied_volatility import VolatilityEstimator


@dataclass
class StockPosition:
    """Represents a stock position."""
    symbol: str
    quantity: int
    current_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


class IntegratedPortfolio:
    """
    Integrated portfolio containing both stocks and options.
    """

    def __init__(self, name: str = "Integrated Portfolio"):
        """
        Initialize integrated portfolio.

        Parameters:
        -----------
        name : str
            Portfolio name
        """
        self.name = name
        self.stock_positions: List[StockPosition] = []
        self.option_portfolio = OptionPortfolio("Options Component")

    def add_stock_position(self, stock_position: StockPosition):
        """Add a stock position."""
        self.stock_positions.append(stock_position)

    def add_option_position(self, option_position: OptionPosition):
        """Add an option position."""
        self.option_portfolio.add_position(option_position)

    @property
    def total_stock_value(self) -> float:
        """Total value of stock positions."""
        return sum(pos.market_value for pos in self.stock_positions)

    @property
    def total_option_value(self) -> float:
        """Total value of option positions."""
        return self.option_portfolio.total_value

    @property
    def total_portfolio_value(self) -> float:
        """Total portfolio value."""
        return self.total_stock_value + self.total_option_value

    def get_symbols(self) -> List[str]:
        """Get all unique symbols in the portfolio."""
        symbols = set()

        # Add stock symbols
        for pos in self.stock_positions:
            symbols.add(pos.symbol)

        # Add option underlying symbols
        for pos in self.option_portfolio.all_positions:
            symbols.add(pos.symbol)

        return list(symbols)


class IntegratedRiskManager:
    """
    Comprehensive risk management for integrated stock-option portfolios.
    """

    def __init__(self, portfolio: IntegratedPortfolio):
        """
        Initialize integrated risk manager.

        Parameters:
        -----------
        portfolio : IntegratedPortfolio
            The integrated portfolio to manage
        """
        self.portfolio = portfolio
        self.vol_estimator = VolatilityEstimator()

        # Initialize component analyzers
        if self.portfolio.option_portfolio.all_positions:
            self.greeks_analyzer = GreeksRiskAnalyzer(self.portfolio.option_portfolio)
            self.option_var_calc = OptionVaRCalculator(self.portfolio.option_portfolio)
            self.stress_tester = OptionStressTester(self.portfolio.option_portfolio)
        else:
            self.greeks_analyzer = None
            self.option_var_calc = None
            self.stress_tester = None

    def calculate_portfolio_exposures(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate comprehensive portfolio exposures.

        Parameters:
        -----------
        market_data : dict
            Market data for each symbol

        Returns:
        --------
        dict : Portfolio exposures
        """
        symbols = self.portfolio.get_symbols()
        exposures = {}

        for symbol in symbols:
            if symbol not in market_data:
                continue

            # Stock exposure
            stock_exposure = 0
            stock_positions = [pos for pos in self.portfolio.stock_positions if pos.symbol == symbol]
            for pos in stock_positions:
                stock_exposure += pos.market_value

            # Option delta exposure
            option_delta_exposure = 0
            option_positions = [pos for pos in self.portfolio.option_portfolio.all_positions
                             if pos.symbol == symbol]
            for pos in option_positions:
                # Delta exposure is equivalent stock exposure
                delta = pos.greeks['delta']
                underlying_value = pos.underlying_price * abs(pos.quantity) * delta
                option_delta_exposure += underlying_value

            # Net exposure
            net_exposure = stock_exposure + option_delta_exposure

            # Calculate volatility
            price_data = market_data[symbol]['Close']
            vol = self.vol_estimator.estimate_volatility(price_data)

            exposures[symbol] = {
                'stock_exposure': stock_exposure,
                'option_delta_exposure': option_delta_exposure,
                'net_exposure': net_exposure,
                'volatility': vol,
                'current_price': price_data.iloc[-1] if not price_data.empty else 0,
                'exposure_percentage': (net_exposure / self.portfolio.total_portfolio_value) * 100
                                     if self.portfolio.total_portfolio_value != 0 else 0
            }

        return exposures

    def integrated_var_analysis(self,
                               market_data: Dict[str, pd.DataFrame],
                               confidence_levels: List[float] = [0.01, 0.05, 0.10],
                               time_horizon: int = 1) -> Dict:
        """
        Calculate integrated VaR considering both stocks and options.

        Parameters:
        -----------
        market_data : dict
            Market data for each symbol
        confidence_levels : list
            Confidence levels for VaR calculation
        time_horizon : int
            Time horizon in days

        Returns:
        --------
        dict : Integrated VaR analysis
        """
        symbols = self.portfolio.get_symbols()
        var_results = {}

        # Calculate returns for stock positions
        stock_returns_data = {}
        for symbol in symbols:
            if symbol in market_data:
                returns = market_data[symbol]['Close'].pct_change().dropna()
                if len(returns) > 0:
                    stock_returns_data[symbol] = returns

        # Create combined returns matrix
        if stock_returns_data:
            returns_df = pd.DataFrame(stock_returns_data).dropna()
        else:
            returns_df = pd.DataFrame()

        for confidence in confidence_levels:
            # Stock component VaR
            stock_var = 0
            if not returns_df.empty and self.portfolio.stock_positions:
                # Calculate portfolio weights for stocks
                stock_values = {}
                total_stock_value = self.portfolio.total_stock_value

                for pos in self.portfolio.stock_positions:
                    if pos.symbol in stock_values:
                        stock_values[pos.symbol] += pos.market_value
                    else:
                        stock_values[pos.symbol] = pos.market_value

                # Create weight vector
                weights = []
                symbols_with_data = []
                for symbol in returns_df.columns:
                    if symbol in stock_values:
                        weights.append(stock_values[symbol] / total_stock_value)
                        symbols_with_data.append(symbol)

                if weights and symbols_with_data:
                    weights = np.array(weights)
                    portfolio_returns = (returns_df[symbols_with_data] * weights).sum(axis=1)

                    var_model = VaRModel()
                    stock_var_relative = var_model.historical_var(portfolio_returns, confidence)
                    stock_var = stock_var_relative * total_stock_value

            # Option component VaR
            option_var = 0
            if self.option_var_calc and self.portfolio.option_portfolio.all_positions:
                # Get volatilities for option VaR calculation
                underlying_vols = {}
                for symbol in symbols:
                    if symbol in market_data:
                        price_data = market_data[symbol]['Close']
                        vol = self.vol_estimator.estimate_volatility(price_data)
                        underlying_vols[symbol] = vol

                # Calculate option VaR using Delta-Gamma method
                option_var_result = self.option_var_calc.delta_gamma_var(
                    confidence_level=confidence,
                    time_horizon=time_horizon,
                    underlying_volatilities=underlying_vols
                )
                option_var = option_var_result['var']

            # Combined VaR (simplified - assumes some correlation)
            # In practice, would use full Monte Carlo simulation
            correlation_factor = 0.7  # Assumed correlation between stock and option components
            combined_var = np.sqrt(stock_var**2 + option_var**2 +
                                 2 * correlation_factor * stock_var * option_var)

            var_results[f'VaR_{int(confidence*100)}%'] = {
                'stock_var': stock_var,
                'option_var': option_var,
                'combined_var': combined_var,
                'correlation_factor': correlation_factor
            }

        return {
            'var_results': var_results,
            'time_horizon_days': time_horizon,
            'total_portfolio_value': self.portfolio.total_portfolio_value,
            'stock_component_value': self.portfolio.total_stock_value,
            'option_component_value': self.portfolio.total_option_value
        }

    def greeks_impact_analysis(self) -> Dict:
        """
        Analyze the impact of option Greeks on overall portfolio.

        Returns:
        --------
        dict : Greeks impact analysis
        """
        if not self.greeks_analyzer:
            return {'error': 'No option positions in portfolio'}

        greeks_report = self.greeks_analyzer.comprehensive_greeks_report()
        portfolio_value = self.portfolio.total_portfolio_value

        # Calculate Greeks as percentage of portfolio value
        portfolio_greeks = greeks_report['portfolio_summary']

        greeks_impact = {}
        for greek, value in portfolio_greeks.items():
            # Impact scenarios
            if greek == 'delta':
                # 1% price move impact
                impact_1pct = abs(value * 0.01)
                greeks_impact[greek] = {
                    'absolute_exposure': value,
                    'impact_1pct_move': impact_1pct,
                    'impact_1pct_percentage': (impact_1pct / portfolio_value) * 100
                                            if portfolio_value != 0 else 0
                }
            elif greek == 'gamma':
                # Gamma shows how delta changes
                # For 1% price move, delta changes by gamma * 1%
                delta_change = value * 0.01
                greeks_impact[greek] = {
                    'absolute_exposure': value,
                    'delta_change_1pct': delta_change,
                    'high_gamma_warning': abs(value) > 10
                }
            elif greek == 'vega':
                # 1% volatility increase impact
                impact_1pct_vol = value * 0.01
                greeks_impact[greek] = {
                    'absolute_exposure': value,
                    'impact_1pct_vol': impact_1pct_vol,
                    'impact_1pct_vol_percentage': (impact_1pct_vol / portfolio_value) * 100
                                                if portfolio_value != 0 else 0
                }
            elif greek == 'theta':
                # Daily time decay
                daily_decay = value / 365
                greeks_impact[greek] = {
                    'absolute_exposure': value,
                    'daily_decay': daily_decay,
                    'daily_decay_percentage': (abs(daily_decay) / portfolio_value) * 100
                                            if portfolio_value != 0 else 0
                }
            elif greek == 'rho':
                # 1% interest rate change impact
                impact_1pct_rate = value * 0.01
                greeks_impact[greek] = {
                    'absolute_exposure': value,
                    'impact_1pct_rate': impact_1pct_rate,
                    'impact_1pct_rate_percentage': (abs(impact_1pct_rate) / portfolio_value) * 100
                                                 if portfolio_value != 0 else 0
                }

        return {
            'greeks_impact': greeks_impact,
            'portfolio_value': portfolio_value,
            'option_component_percentage': (self.portfolio.total_option_value / portfolio_value) * 100
                                         if portfolio_value != 0 else 0,
            'detailed_greeks_report': greeks_report
        }

    def comprehensive_stress_test(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run comprehensive stress test on the integrated portfolio.

        Parameters:
        -----------
        market_data : dict
            Market data for each symbol

        Returns:
        --------
        dict : Comprehensive stress test results
        """
        scenarios = {}

        # Market crash scenarios - affect both stocks and options
        for crash_size in [0.10, 0.20, 0.30]:
            scenario_name = f'market_crash_{int(crash_size*100)}pct'

            # Stock impact
            stock_impact = 0
            for pos in self.portfolio.stock_positions:
                price_drop = pos.current_price * crash_size
                stock_impact -= pos.quantity * price_drop

            # Option impact
            option_impact = 0
            if self.stress_tester:
                option_stress = self.stress_tester.market_crash_scenario(crash_size)
                option_impact = option_stress['total_pnl']

            total_impact = stock_impact + option_impact

            scenarios[scenario_name] = {
                'scenario': f'Market Crash -{int(crash_size*100)}%',
                'stock_impact': stock_impact,
                'option_impact': option_impact,
                'total_impact': total_impact,
                'impact_percentage': (total_impact / self.portfolio.total_portfolio_value) * 100
                                   if self.portfolio.total_portfolio_value != 0 else 0
            }

        # Volatility scenarios - mainly affect options
        if self.stress_tester:
            for vol_shock in [0.05, 0.10, 0.15, -0.05]:
                sign = "up" if vol_shock > 0 else "down"
                scenario_name = f'vol_shock_{sign}_{int(abs(vol_shock)*100)}pct'

                option_stress = self.stress_tester.volatility_shock_scenario(vol_shock)
                option_impact = option_stress['total_pnl']

                scenarios[scenario_name] = {
                    'scenario': f'Volatility Shock {vol_shock:+.1%}',
                    'stock_impact': 0,  # Stocks not directly affected by vol
                    'option_impact': option_impact,
                    'total_impact': option_impact,
                    'impact_percentage': (option_impact / self.portfolio.total_portfolio_value) * 100
                                       if self.portfolio.total_portfolio_value != 0 else 0
                }

        # Interest rate scenarios
        for rate_shock in [0.005, 0.01, 0.02, -0.005]:  # 50bp, 100bp, 200bp
            scenario_name = f'rate_shock_{int(rate_shock*10000)}bp'

            # Options affected by rho
            option_impact = 0
            if self.portfolio.option_portfolio.all_positions:
                portfolio_rho = self.portfolio.option_portfolio.portfolio_greeks['rho']
                option_impact = portfolio_rho * rate_shock

            # Stocks indirectly affected (discount rate changes)
            # Simplified: assume 10% of portfolio value sensitivity to rates
            stock_impact = -self.portfolio.total_stock_value * 0.10 * rate_shock

            total_impact = stock_impact + option_impact

            scenarios[scenario_name] = {
                'scenario': f'Interest Rate Shock {rate_shock:+.2%}',
                'stock_impact': stock_impact,
                'option_impact': option_impact,
                'total_impact': total_impact,
                'impact_percentage': (total_impact / self.portfolio.total_portfolio_value) * 100
                                   if self.portfolio.total_portfolio_value != 0 else 0
            }

        # Find worst and best case scenarios
        worst_case = min(scenarios.items(), key=lambda x: x[1]['total_impact'])
        best_case = max(scenarios.items(), key=lambda x: x[1]['total_impact'])

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
            'portfolio_composition': {
                'total_value': self.portfolio.total_portfolio_value,
                'stock_value': self.portfolio.total_stock_value,
                'option_value': self.portfolio.total_option_value,
                'stock_percentage': (self.portfolio.total_stock_value / self.portfolio.total_portfolio_value) * 100
                                  if self.portfolio.total_portfolio_value != 0 else 0,
                'option_percentage': (self.portfolio.total_option_value / self.portfolio.total_portfolio_value) * 100
                                   if self.portfolio.total_portfolio_value != 0 else 0
            }
        }

    def generate_risk_dashboard_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate comprehensive data for risk dashboard visualization.

        Parameters:
        -----------
        market_data : dict
            Market data for each symbol

        Returns:
        --------
        dict : Risk dashboard data
        """
        dashboard_data = {}

        # Portfolio summary
        dashboard_data['portfolio_summary'] = {
            'total_value': self.portfolio.total_portfolio_value,
            'stock_value': self.portfolio.total_stock_value,
            'option_value': self.portfolio.total_option_value,
            'num_stock_positions': len(self.portfolio.stock_positions),
            'num_option_positions': len(self.portfolio.option_portfolio.all_positions),
            'num_symbols': len(self.portfolio.get_symbols())
        }

        # Exposures
        dashboard_data['exposures'] = self.calculate_portfolio_exposures(market_data)

        # VaR analysis
        dashboard_data['var_analysis'] = self.integrated_var_analysis(market_data)

        # Greeks impact
        dashboard_data['greeks_analysis'] = self.greeks_impact_analysis()

        # Stress tests
        dashboard_data['stress_tests'] = self.comprehensive_stress_test(market_data)

        # Risk alerts
        dashboard_data['risk_alerts'] = self._generate_risk_alerts(dashboard_data)

        return dashboard_data

    def _generate_risk_alerts(self, dashboard_data: Dict) -> List[Dict]:
        """Generate risk alerts based on analysis."""
        alerts = []

        # High concentration alert
        exposures = dashboard_data.get('exposures', {})
        for symbol, data in exposures.items():
            if data['exposure_percentage'] > 25:  # More than 25% in single symbol
                alerts.append({
                    'level': 'HIGH',
                    'type': 'Concentration Risk',
                    'message': f'High exposure to {symbol}: {data["exposure_percentage"]:.1f}%',
                    'symbol': symbol
                })

        # Greeks alerts
        greeks_analysis = dashboard_data.get('greeks_analysis', {})
        if 'greeks_impact' in greeks_analysis:
            greeks_impact = greeks_analysis['greeks_impact']

            # High delta alert
            if 'delta' in greeks_impact and greeks_impact['delta']['impact_1pct_percentage'] > 5:
                alerts.append({
                    'level': 'MEDIUM',
                    'type': 'Delta Risk',
                    'message': f'High delta exposure: {greeks_impact["delta"]["impact_1pct_percentage"]:.1f}% for 1% move'
                })

            # High gamma alert
            if 'gamma' in greeks_impact and greeks_impact['gamma'].get('high_gamma_warning', False):
                alerts.append({
                    'level': 'MEDIUM',
                    'type': 'Gamma Risk',
                    'message': 'High gamma exposure detected - delta will change rapidly'
                })

            # High vega alert
            if 'vega' in greeks_impact and greeks_impact['vega']['impact_1pct_vol_percentage'] > 3:
                alerts.append({
                    'level': 'MEDIUM',
                    'type': 'Vega Risk',
                    'message': f'High vega exposure: {greeks_impact["vega"]["impact_1pct_vol_percentage"]:.1f}% for 1% vol change'
                })

        # Stress test alerts
        stress_tests = dashboard_data.get('stress_tests', {})
        if 'worst_case' in stress_tests:
            worst_impact = stress_tests['worst_case']['results']['impact_percentage']
            if abs(worst_impact) > 20:  # More than 20% loss in worst case
                alerts.append({
                    'level': 'HIGH',
                    'type': 'Stress Test',
                    'message': f'Worst case scenario: {worst_impact:.1f}% portfolio impact',
                    'scenario': stress_tests['worst_case']['scenario_name']
                })

        return alerts