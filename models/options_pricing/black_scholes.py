"""
Black-Scholes Option Pricing Model

Implementation of the classic Black-Scholes model for European option pricing.
Follows the mathematical framework: Pure Math → Applied Math → Financial Models

Mathematical Foundation:
- Pure Math: Normal distributions, exponential functions
- Applied Math: Geometric Brownian motion + Black-Scholes PDE
- Financial Model: Analytical option pricing formulas
"""
import numpy as np
from scipy.stats import norm
from typing import Union, Tuple, Optional
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .base_model import BaseModel, ModelParameters


@dataclass
class BSParameters(ModelParameters):
    """Black-Scholes model parameters"""
    sigma: float = 0.2  # Volatility

    def __post_init__(self):
        """Validate BS-specific parameters"""
        super().__post_init__()
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")


class BlackScholesModel(BaseModel):
    """Black-Scholes option pricing and Greeks calculation"""

    def __init__(self, params: BSParameters):
        super().__init__(params)

    def _d1_d2(self) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters"""
        S0, K, T, r, sigma, q = (self.params.S0, self.params.K, self.params.T,
                                 self.params.r, self.params.sigma, self.params.q)

        d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        return d1, d2

    def call_price(self) -> float:
        """European call option price"""
        S0, K, T, r, q = self.params.S0, self.params.K, self.params.T, self.params.r, self.params.q
        d1, d2 = self._d1_d2()

        call = S0 * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        return call

    def put_price(self) -> float:
        """European put option price"""
        S0, K, T, r, q = self.params.S0, self.params.K, self.params.T, self.params.r, self.params.q
        d1, d2 = self._d1_d2()

        put = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * np.exp(-q*T) * norm.cdf(-d1)
        return put

    def delta(self, option_type: str = 'call') -> float:
        """Option delta"""
        d1, _ = self._d1_d2()
        if option_type.lower() == 'call':
            return np.exp(-self.params.q * self.params.T) * norm.cdf(d1)
        elif option_type.lower() == 'put':
            return np.exp(-self.params.q * self.params.T) * (norm.cdf(d1) - 1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

    def call_delta(self) -> float:
        """Call option delta (deprecated - use delta('call'))"""
        return self.delta('call')

    def put_delta(self) -> float:
        """Put option delta (deprecated - use delta('put'))"""
        return self.delta('put')

    def gamma(self) -> float:
        """Option gamma (same for calls and puts)"""
        S0, T, sigma, q = self.params.S0, self.params.T, self.params.sigma, self.params.q
        d1, _ = self._d1_d2()

        return np.exp(-q*T) * norm.pdf(d1) / (S0 * sigma * np.sqrt(T))

    def vega(self) -> float:
        """Option vega (same for calls and puts)"""
        S0, T, q = self.params.S0, self.params.T, self.params.q
        d1, _ = self._d1_d2()

        return S0 * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)

    def theta(self, option_type: str = 'call') -> float:
        """Option theta"""
        S0, K, T, r, sigma, q = (self.params.S0, self.params.K, self.params.T,
                               self.params.r, self.params.sigma, self.params.q)
        d1, d2 = self._d1_d2()

        theta1 = -(S0 * np.exp(-q*T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type.lower() == 'call':
            theta2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
            theta3 = q * S0 * np.exp(-q*T) * norm.cdf(d1)
        elif option_type.lower() == 'put':
            theta2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
            theta3 = -q * S0 * np.exp(-q*T) * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return theta1 + theta2 + theta3

    def call_theta(self) -> float:
        """Call option theta (deprecated - use theta('call'))"""
        return self.theta('call')

    def put_theta(self) -> float:
        """Put option theta (deprecated - use theta('put'))"""
        return self.theta('put')

    def rho(self, option_type: str = 'call') -> float:
        """Option rho"""
        K, T, r = self.params.K, self.params.T, self.params.r
        _, d2 = self._d1_d2()

        if option_type.lower() == 'call':
            return K * T * np.exp(-r*T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            return -K * T * np.exp(-r*T) * norm.cdf(-d2)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

    def call_rho(self) -> float:
        """Call option rho (deprecated - use rho('call'))"""
        return self.rho('call')

    def put_rho(self) -> float:
        """Put option rho (deprecated - use rho('put'))"""
        return self.rho('put')

    def implied_volatility(self, market_price: float, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method"""

        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")

        sigma = 0.3

        for i in range(max_iterations):
            self.params.sigma = sigma

            if option_type.lower() == 'call':
                theoretical_price = self.call_price()
            else:
                theoretical_price = self.put_price()

            vega_value = self.vega()

            price_diff = theoretical_price - market_price
            if abs(price_diff) < tolerance:
                return sigma

            if vega_value == 0:
                raise ValueError("Vega is zero, cannot calculate implied volatility")

            sigma = sigma - price_diff / vega_value
            sigma = max(sigma, 1e-6)

        raise ValueError(f"Implied volatility did not converge after {max_iterations} iterations")

    def monte_carlo_price(self, n_paths: int = 100000, n_steps: int = 252,
                         option_type: str = 'call', seed: Optional[int] = None) -> float:
        """Monte Carlo pricing using geometric Brownian motion"""
        if seed is not None:
            np.random.seed(seed)

        S0, K, T, r, sigma, q = (self.params.S0, self.params.K, self.params.T,
                                self.params.r, self.params.sigma, self.params.q)

        dt = T / n_steps
        S = np.full(n_paths, S0)

        # Generate paths using exact GBM simulation
        for _ in range(n_steps):
            Z = np.random.standard_normal(n_paths)
            S = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return np.exp(-r * T) * np.mean(payoffs)


def price_european_option(S0: float, K: float, T: float, r: float, sigma: float,
                         option_type: str = 'call', q: float = 0.0) -> float:
    """Convenience function to price European options"""
    params = BSParameters(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
    model = BlackScholesModel(params)

    if option_type.lower() == 'call':
        return model.call_price()
    elif option_type.lower() == 'put':
        return model.put_price()
    else:
        raise ValueError("Option type must be 'call' or 'put'")


def calculate_greeks(S0: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = 'call', q: float = 0.0) -> dict:
    """Calculate all Greeks for an option"""
    params = BSParameters(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
    model = BlackScholesModel(params)

    greeks = {
        'price': model.call_price() if option_type.lower() == 'call' else model.put_price(),
        'delta': model.call_delta() if option_type.lower() == 'call' else model.put_delta(),
        'gamma': model.gamma(),
        'vega': model.vega(),
        'theta': model.call_theta() if option_type.lower() == 'call' else model.put_theta(),
        'rho': model.call_rho() if option_type.lower() == 'call' else model.put_rho()
    }

    return greeks


# Register with the model factory
from .base_model import ModelFactory
ModelFactory.register_model('black-scholes', BlackScholesModel)
ModelFactory.register_model('bs', BlackScholesModel)


if __name__ == "__main__":
    params = BSParameters(S0=100, K=105, T=0.25, r=0.05, sigma=0.2, q=0.02)
    model = BlackScholesModel(params)

    print("Black-Scholes Model Results:")
    print(f"Call Price: {model.call_price():.4f}")
    print(f"Put Price: {model.put_price():.4f}")
    print(f"Call Delta: {model.delta('call'):.4f}")
    print(f"Put Delta: {model.delta('put'):.4f}")
    print(f"Gamma: {model.gamma():.4f}")
    print(f"Vega: {model.vega():.4f}")
    print(f"Call Theta: {model.theta('call'):.4f}")
    print(f"Put Theta: {model.theta('put'):.4f}")
    print(f"Call Rho: {model.rho('call'):.4f}")
    print(f"Put Rho: {model.rho('put'):.4f}")

    print("\nMonte Carlo Comparison:")
    print(f"Analytic Call: {model.call_price():.4f}")
    print(f"MC Call: {model.monte_carlo_price('call', seed=42):.4f}")

    print("\nGreeks Dictionary:")
    print(model.greeks('call'))