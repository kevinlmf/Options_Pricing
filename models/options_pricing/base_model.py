"""
Base classes for all pricing models.

Provides abstract interfaces that all specific models should implement,
following the mathematical framework outlined in the theory documentation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import numpy as np


@dataclass
class ModelParameters(ABC):
    """Base class for all model parameters"""
    S0: float = 100.0  # Current asset price
    K: float = 100.0   # Strike price
    T: float = 1.0     # Time to maturity
    r: float = 0.05    # Risk-free rate
    q: float = 0.0     # Dividend yield

    def __post_init__(self):
        """Validate parameters after initialization"""
        if self.S0 <= 0:
            raise ValueError("Current asset price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.r < 0:
            raise ValueError("Risk-free rate cannot be negative")
        if self.q < 0:
            raise ValueError("Dividend yield cannot be negative")


class BaseModel(ABC):
    """
    Abstract base class for all pricing models.

    This class defines the interface that all specific models should implement.
    It follows the mathematical progression:
    Pure Math → Applied Math → Financial Models
    """

    def __init__(self, params: ModelParameters):
        """Initialize model with parameters"""
        self.params = params
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate model parameters"""
        # Base validation is handled in ModelParameters.__post_init__
        pass

    @abstractmethod
    def call_price(self) -> float:
        """Calculate European call option price"""
        pass

    @abstractmethod
    def put_price(self) -> float:
        """Calculate European put option price"""
        pass

    def option_price(self, option_type: str = 'call') -> float:
        """Generic option pricing method"""
        if option_type.lower() == 'call':
            return self.call_price()
        elif option_type.lower() == 'put':
            return self.put_price()
        else:
            raise ValueError("Option type must be 'call' or 'put'")

    @abstractmethod
    def delta(self, option_type: str = 'call') -> float:
        """Calculate option delta"""
        pass

    @abstractmethod
    def gamma(self) -> float:
        """Calculate option gamma"""
        pass

    @abstractmethod
    def vega(self) -> float:
        """Calculate option vega"""
        pass

    @abstractmethod
    def theta(self, option_type: str = 'call') -> float:
        """Calculate option theta"""
        pass

    @abstractmethod
    def rho(self, option_type: str = 'call') -> float:
        """Calculate option rho"""
        pass

    def greeks(self, option_type: str = 'call') -> Dict[str, float]:
        """Calculate all Greeks for the option"""
        return {
            'price': self.option_price(option_type),
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type)
        }

    @abstractmethod
    def implied_volatility(self, market_price: float, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """Calculate implied volatility from market price"""
        pass

    def monte_carlo_price(self, n_paths: int = 100000, n_steps: int = 252,
                         option_type: str = 'call', seed: Optional[int] = None) -> float:
        """
        Monte Carlo pricing - default implementation using geometric Brownian motion.
        Specific models can override this for more sophisticated path generation.
        """
        if seed is not None:
            np.random.seed(seed)

        # This is a basic implementation - models should override with their specific dynamics
        dt = self.params.T / n_steps
        S = np.full(n_paths, self.params.S0)

        # Basic geometric Brownian motion (models should override)
        for _ in range(n_steps):
            Z = np.random.standard_normal(n_paths)
            S = S * np.exp((self.params.r - self.params.q - 0.5 * 0.2**2) * dt +
                          0.2 * np.sqrt(dt) * Z)  # Using 20% vol as placeholder

        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S - self.params.K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(self.params.K - S, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return np.exp(-self.params.r * self.params.T) * np.mean(payoffs)

    def __str__(self) -> str:
        """String representation of the model"""
        return f"{self.__class__.__name__}(S0={self.params.S0}, K={self.params.K}, T={self.params.T}, r={self.params.r})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()


class ModelFactory:
    """Factory class for creating pricing models"""

    _models = {}

    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a new model type"""
        cls._models[name.lower()] = model_class

    @classmethod
    def create_model(cls, name: str, params: ModelParameters) -> BaseModel:
        """Create a model instance by name"""
        if name.lower() not in cls._models:
            raise ValueError(f"Unknown model type: {name}")
        return cls._models[name.lower()](params)

    @classmethod
    def available_models(cls) -> list:
        """List all available model types"""
        return list(cls._models.keys())