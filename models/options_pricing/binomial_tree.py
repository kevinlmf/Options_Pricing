"""
Binomial Tree Model for Option Pricing

Implementation of the Cox-Ross-Rubinstein binomial tree model.
Follows the mathematical framework: Pure Math → Applied Math → Financial Models

Mathematical Foundation:
- Pure Math: Discrete probability, recursion, tree structures
- Applied Math: Discrete approximation of continuous processes
- Financial Model: American and European option pricing via backward induction
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from .base_model import BaseModel, ModelParameters


@dataclass
class BinomialParameters(ModelParameters):
    """Binomial tree model parameters"""
    sigma: float = 0.2  # Volatility
    n_steps: int = 100  # Number of time steps

    def __post_init__(self):
        """Validate binomial-specific parameters"""
        super().__post_init__()
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive")


class BinomialTreeModel(BaseModel):
    """
    Binomial tree model for American and European options.

    Uses Cox-Ross-Rubinstein parameterization:
    - u = exp(sigma * sqrt(dt))
    - d = 1/u
    - p = (exp(r*dt) - d) / (u - d)
    """

    def __init__(self, params: BinomialParameters):
        super().__init__(params)
        self._setup_tree_parameters()

    def _setup_tree_parameters(self):
        """Calculate tree parameters"""
        self.dt = self.params.T / self.params.n_steps
        self.u = np.exp(self.params.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u

        # Risk-neutral probability
        exp_r_dt = np.exp(self.params.r * self.dt)
        self.p = (exp_r_dt - self.d) / (self.u - self.d)

        if not (0 < self.p < 1):
            raise ValueError("Risk-neutral probability must be between 0 and 1")

    def _build_stock_tree(self) -> np.ndarray:
        """Build stock price tree"""
        n = self.params.n_steps
        stock_tree = np.zeros((n + 1, n + 1))

        for i in range(n + 1):
            for j in range(i + 1):
                stock_tree[j, i] = self.params.S0 * (self.u ** (i - j)) * (self.d ** j)

        return stock_tree

    def _european_option_value(self, option_type: str = 'call') -> float:
        """Calculate European option value using binomial tree"""
        n = self.params.n_steps
        stock_tree = self._build_stock_tree()

        # Initialize option value tree
        option_tree = np.zeros((n + 1, n + 1))

        # Terminal payoffs
        for j in range(n + 1):
            S_T = stock_tree[j, n]
            if option_type.lower() == 'call':
                option_tree[j, n] = max(S_T - self.params.K, 0)
            else:
                option_tree[j, n] = max(self.params.K - S_T, 0)

        # Backward induction
        discount = np.exp(-self.params.r * self.dt)
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = discount * (self.p * option_tree[j, i + 1] +
                                               (1 - self.p) * option_tree[j + 1, i + 1])

        return option_tree[0, 0]

    def _american_option_value(self, option_type: str = 'call') -> float:
        """Calculate American option value using binomial tree"""
        n = self.params.n_steps
        stock_tree = self._build_stock_tree()

        # Initialize option value tree
        option_tree = np.zeros((n + 1, n + 1))

        # Terminal payoffs
        for j in range(n + 1):
            S_T = stock_tree[j, n]
            if option_type.lower() == 'call':
                option_tree[j, n] = max(S_T - self.params.K, 0)
            else:
                option_tree[j, n] = max(self.params.K - S_T, 0)

        # Backward induction with early exercise
        discount = np.exp(-self.params.r * self.dt)
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation = discount * (self.p * option_tree[j, i + 1] +
                                         (1 - self.p) * option_tree[j + 1, i + 1])

                # Intrinsic value (early exercise)
                S = stock_tree[j, i]
                if option_type.lower() == 'call':
                    intrinsic = max(S - self.params.K, 0)
                else:
                    intrinsic = max(self.params.K - S, 0)

                # American option value is max of continuation and exercise
                option_tree[j, i] = max(continuation, intrinsic)

        return option_tree[0, 0]

    def call_price(self, american: bool = False) -> float:
        """European or American call option price"""
        if american:
            return self._american_option_value('call')
        else:
            return self._european_option_value('call')

    def put_price(self, american: bool = False) -> float:
        """European or American put option price"""
        if american:
            return self._american_option_value('put')
        else:
            return self._european_option_value('put')

    def option_price(self, option_type: str = 'call', american: bool = False) -> float:
        """Generic option pricing method"""
        if option_type.lower() == 'call':
            return self.call_price(american)
        elif option_type.lower() == 'put':
            return self.put_price(american)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

    def delta(self, option_type: str = 'call', american: bool = False) -> float:
        """Calculate delta using finite difference"""
        # Build stock tree for delta calculation
        stock_tree = self._build_stock_tree()

        if american:
            # Use American pricing
            up_value = self._get_node_value(0, 1, option_type, american=True)
            down_value = self._get_node_value(1, 1, option_type, american=True)
        else:
            # Use European pricing
            up_value = self._get_node_value(0, 1, option_type, american=False)
            down_value = self._get_node_value(1, 1, option_type, american=False)

        dS = stock_tree[0, 1] - stock_tree[1, 1]
        return (up_value - down_value) / dS if dS != 0 else 0

    def _get_node_value(self, j: int, i: int, option_type: str, american: bool = False) -> float:
        """Get option value at specific tree node"""
        # This is a simplified implementation - would need full tree for accuracy
        # For now, approximate using shifted parameters
        shifted_params = BinomialParameters(
            S0=self.params.S0 * (self.u if j == 0 else self.d),
            K=self.params.K,
            T=self.params.T - i * self.dt,
            r=self.params.r,
            sigma=self.params.sigma,
            n_steps=max(1, self.params.n_steps - i)
        )
        temp_model = BinomialTreeModel(shifted_params)
        return temp_model.option_price(option_type, american)

    def gamma(self) -> float:
        """Gamma calculation - simplified approximation"""
        # This would require more sophisticated tree analysis
        # For now, return 0 as placeholder
        return 0.0

    def vega(self) -> float:
        """Vega calculation using finite difference"""
        vol_shift = 0.01

        # Shift volatility up
        params_up = BinomialParameters(
            S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r,
            sigma=self.params.sigma + vol_shift, n_steps=self.params.n_steps
        )
        model_up = BinomialTreeModel(params_up)

        # Shift volatility down
        params_down = BinomialParameters(
            S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r,
            sigma=max(0.001, self.params.sigma - vol_shift), n_steps=self.params.n_steps
        )
        model_down = BinomialTreeModel(params_down)

        # Use call option for vega calculation
        return (model_up.call_price() - model_down.call_price()) / (2 * vol_shift) / 100

    def theta(self, option_type: str = 'call') -> float:
        """Theta calculation using finite difference"""
        time_shift = min(1/365, self.params.T * 0.01)  # 1 day or 1% of time to expiry

        # Shift time to expiry down
        params_shifted = BinomialParameters(
            S0=self.params.S0, K=self.params.K,
            T=max(0.001, self.params.T - time_shift),
            r=self.params.r, sigma=self.params.sigma, n_steps=self.params.n_steps
        )
        model_shifted = BinomialTreeModel(params_shifted)

        current_price = self.option_price(option_type)
        shifted_price = model_shifted.option_price(option_type)

        return (shifted_price - current_price) / time_shift

    def rho(self, option_type: str = 'call') -> float:
        """Rho calculation using finite difference"""
        rate_shift = 0.0001  # 1 basis point

        # Shift rate up
        params_up = BinomialParameters(
            S0=self.params.S0, K=self.params.K, T=self.params.T,
            r=self.params.r + rate_shift, sigma=self.params.sigma, n_steps=self.params.n_steps
        )
        model_up = BinomialTreeModel(params_up)

        # Shift rate down
        params_down = BinomialParameters(
            S0=self.params.S0, K=self.params.K, T=self.params.T,
            r=self.params.r - rate_shift, sigma=self.params.sigma, n_steps=self.params.n_steps
        )
        model_down = BinomialTreeModel(params_down)

        return (model_up.option_price(option_type) - model_down.option_price(option_type)) / (2 * rate_shift) / 100

    def implied_volatility(self, market_price: float, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        sigma = 0.3  # Initial guess

        for i in range(max_iterations):
            # Create temporary parameters with current sigma
            temp_params = BinomialParameters(
                S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r,
                sigma=sigma, n_steps=self.params.n_steps
            )
            temp_model = BinomialTreeModel(temp_params)

            model_price = temp_model.option_price(option_type)
            price_diff = model_price - market_price

            if abs(price_diff) < tolerance:
                return sigma

            # Calculate vega for Newton-Raphson update
            vega_value = temp_model.vega() * 100  # Convert from % to decimal

            if abs(vega_value) < 1e-10:
                raise ValueError("Vega too small for implied volatility calculation")

            sigma = sigma - price_diff / vega_value
            sigma = max(sigma, 1e-6)  # Ensure positive volatility

        raise ValueError(f"Implied volatility did not converge after {max_iterations} iterations")

    def monte_carlo_price(self, n_paths: int = 100000, n_steps: int = 252,
                         option_type: str = 'call', seed: Optional[int] = None) -> float:
        """
        Monte Carlo simulation - uses the binomial parameters but simulates continuous paths.
        This provides a comparison between discrete tree and continuous approximation.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = self.params.T / n_steps
        S = np.full(n_paths, self.params.S0)

        # Use geometric Brownian motion for continuous approximation
        for _ in range(n_steps):
            Z = np.random.standard_normal(n_paths)
            S = S * np.exp((self.params.r - self.params.q - 0.5 * self.params.sigma**2) * dt +
                          self.params.sigma * np.sqrt(dt) * Z)

        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S - self.params.K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(self.params.K - S, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return np.exp(-self.params.r * self.params.T) * np.mean(payoffs)


# Register with the model factory
from .base_model import ModelFactory
ModelFactory.register_model('binomial', BinomialTreeModel)
ModelFactory.register_model('binomial-tree', BinomialTreeModel)


if __name__ == "__main__":
    params = BinomialParameters(S0=100, K=105, T=0.25, r=0.05, sigma=0.2, n_steps=100)
    model = BinomialTreeModel(params)

    print("Binomial Tree Model Results:")
    print(f"European Call: {model.call_price(american=False):.4f}")
    print(f"European Put: {model.put_price(american=False):.4f}")
    print(f"American Call: {model.call_price(american=True):.4f}")
    print(f"American Put: {model.put_price(american=True):.4f}")

    print(f"\nGreeks:")
    print(f"Delta (Call): {model.delta('call'):.4f}")
    print(f"Vega: {model.vega():.4f}")
    print(f"Theta (Call): {model.theta('call'):.4f}")
    print(f"Rho (Call): {model.rho('call'):.4f}")

    print(f"\nComparison with Monte Carlo:")
    print(f"Binomial European Call: {model.call_price(american=False):.4f}")
    print(f"MC Call: {model.monte_carlo_price('call', seed=42):.4f}")