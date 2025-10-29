"""
Heston Stochastic Volatility Model

Implementation of the Heston stochastic volatility model for option pricing.
Follows the mathematical framework: Pure Math → Applied Math → Financial Models

Mathematical Foundation:
- Pure Math: Complex analysis, characteristic functions, Fourier transforms
- Applied Math: 2D stochastic differential equations, numerical integration
- Financial Model: Semi-analytical option pricing via characteristic function methods
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm
from dataclasses import dataclass
from typing import Tuple, Optional
import cmath
from .base_model import BaseModel, ModelParameters

# Try to import C++ accelerated module
try:
    import heston_cpp
    HESTON_CPP_AVAILABLE = True
except ImportError:
    HESTON_CPP_AVAILABLE = False
    import warnings
    warnings.warn(
        "C++ accelerated Heston module not available. "
        "Build it with: cd cpp_accelerators && make. "
        "Falling back to pure Python implementation (slower)."
    )


@dataclass
class HestonParameters(ModelParameters):
    """Heston model parameters"""
    v0: float = 0.04      # Initial variance
    kappa: float = 2.0    # Mean reversion speed
    theta: float = 0.04   # Long-run variance
    xi: float = 0.3       # Vol of vol
    rho: float = -0.5     # Correlation between price and vol

    def __post_init__(self):
        """Validate Heston-specific parameters"""
        super().__post_init__()
        if self.v0 <= 0:
            raise ValueError("Initial variance must be positive")
        if self.kappa <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if self.theta <= 0:
            raise ValueError("Long-run variance must be positive")
        if self.xi <= 0:
            raise ValueError("Vol of vol must be positive")
        if abs(self.rho) >= 1:
            raise ValueError("Correlation must be between -1 and 1")

        # Check Feller condition
        if 2 * self.kappa * self.theta <= self.xi ** 2:
            import warnings
            warnings.warn("Feller condition (2κθ > ξ²) not satisfied - variance may become negative")


class HestonModel(BaseModel):
    """Heston stochastic volatility model for option pricing"""

    def __init__(self, params: HestonParameters):
        super().__init__(params)

    def characteristic_function(self, u: complex) -> complex:
        """Heston characteristic function"""
        S0, K, T, r, v0, kappa, theta, xi, rho, q = (
            self.params.S0, self.params.K, self.params.T, self.params.r,
            self.params.v0, self.params.kappa, self.params.theta,
            self.params.xi, self.params.rho, self.params.q
        )

        # Calculate d
        d = cmath.sqrt((rho * xi * u * 1j - kappa)**2 + xi**2 * (u * 1j + u**2))

        # Calculate g
        g = (kappa - rho * xi * u * 1j - d) / (kappa - rho * xi * u * 1j + d)

        # Calculate C(u, T)
        C = (r - q) * u * 1j * T + (kappa * theta / xi**2) * (
            (kappa - rho * xi * u * 1j - d) * T - 2 * cmath.log((1 - g * cmath.exp(-d * T)) / (1 - g))
        )

        # Calculate D(u, T)
        D = ((kappa - rho * xi * u * 1j - d) / xi**2) * ((1 - cmath.exp(-d * T)) / (1 - g * cmath.exp(-d * T)))

        # Characteristic function for ln(St/S0) - log return, not log price
        # Note: S0 is already included in the pricing formula, so we don't include ln(S0) here
        cf = cmath.exp(C + D * v0)

        return cf

    def _integrand_P1(self, phi: float) -> float:
        """Integrand for P1 calculation"""
        S0, K = self.params.S0, self.params.K

        cf = self.characteristic_function(phi - 1j)
        # Use ln(K/S0) since CF is for ln(St/S0)
        numerator = cmath.exp(-1j * phi * cmath.log(K / S0)) * cf
        denominator = 1j * phi

        return (numerator / denominator).real

    def _integrand_P2(self, phi: float) -> float:
        """Integrand for P2 calculation"""
        S0, K = self.params.S0, self.params.K

        cf = self.characteristic_function(phi)
        # Use ln(K/S0) since CF is for ln(St/S0)
        numerator = cmath.exp(-1j * phi * cmath.log(K / S0)) * cf
        denominator = 1j * phi

        return (numerator / denominator).real

    def option_price(self, option_type: str = 'call', use_cpp: bool = True) -> float:
        """
        Calculate option price using Heston formula

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        use_cpp : bool
            Use C++ acceleration if available (default: True)
        """
        # Try C++ acceleration first if available and requested
        if use_cpp and HESTON_CPP_AVAILABLE:
            try:
                result = heston_cpp.price_option(
                    S0=self.params.S0,
                    K=self.params.K,
                    T=self.params.T,
                    r=self.params.r,
                    q=self.params.q,
                    v0=self.params.v0,
                    kappa=self.params.kappa,
                    theta=self.params.theta,
                    xi=self.params.xi,
                    rho=self.params.rho,
                    option_type=option_type
                )
                return result['price']
            except Exception as e:
                import warnings
                warnings.warn(f"C++ pricing failed: {e}. Falling back to Python.")

        # Fallback to pure Python implementation
        S0, K, T, r, q = self.params.S0, self.params.K, self.params.T, self.params.r, self.params.q

        try:
            # Calculate P1 and P2 via integration
            P1 = 0.5 + (1/np.pi) * quad(self._integrand_P1, 0, np.inf, limit=1000)[0]
            P2 = 0.5 + (1/np.pi) * quad(self._integrand_P2, 0, np.inf, limit=1000)[0]

            # Call price
            call_price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

            if option_type.lower() == 'call':
                return max(call_price, 0)
            elif option_type.lower() == 'put':
                # Put-call parity
                put_price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
                return max(put_price, 0)
            else:
                raise ValueError("Option type must be 'call' or 'put'")

        except Exception as e:
            print(f"Error in Heston pricing: {e}")
            # Fallback to Black-Scholes approximation
            return self._black_scholes_approximation(option_type)

    def _black_scholes_approximation(self, option_type: str) -> float:
        """Fallback Black-Scholes approximation using initial volatility"""
        from scipy.stats import norm

        S0, K, T, r, q = self.params.S0, self.params.K, self.params.T, self.params.r, self.params.q
        sigma = np.sqrt(self.params.v0)  # Use initial vol as approximation

        d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type.lower() == 'call':
            return S0 * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * np.exp(-q*T) * norm.cdf(-d1)

    def monte_carlo_price(self, n_paths: int = 100000, n_steps: int = 252,
                         option_type: str = 'call', seed: Optional[int] = None) -> float:
        """Monte Carlo pricing using Euler discretization"""
        if seed is not None:
            np.random.seed(seed)

        S0, K, T, r, v0, kappa, theta, xi, rho, q = (
            self.params.S0, self.params.K, self.params.T, self.params.r,
            self.params.v0, self.params.kappa, self.params.theta,
            self.params.xi, self.params.rho, self.params.q
        )

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize arrays
        S = np.full(n_paths, S0)
        v = np.full(n_paths, v0)

        # Generate correlated random numbers
        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)

            # Update volatility (ensure non-negative)
            v_new = v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0)) * sqrt_dt * Z2
            v = np.maximum(v_new, 0)  # Ensure non-negative variance

            # Update stock price
            S = S * np.exp((r - q - 0.5 * v) * dt + np.sqrt(v) * sqrt_dt * Z1)

        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        # Discount back to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        return option_price

    def call_price(self) -> float:
        """European call option price"""
        return self.option_price('call')

    def put_price(self) -> float:
        """European put option price"""
        return self.option_price('put')

    def delta(self, option_type: str = 'call') -> float:
        """Calculate delta using finite difference"""
        shift = 0.01 * self.params.S0  # 1% shift in underlying

        # Shift underlying up
        params_up = HestonParameters(
            S0=self.params.S0 + shift, K=self.params.K, T=self.params.T, r=self.params.r, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_up = HestonModel(params_up)

        # Shift underlying down
        params_down = HestonParameters(
            S0=self.params.S0 - shift, K=self.params.K, T=self.params.T, r=self.params.r, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_down = HestonModel(params_down)

        return (model_up.option_price(option_type) - model_down.option_price(option_type)) / (2 * shift)

    def gamma(self) -> float:
        """Calculate gamma using finite difference"""
        shift = 0.01 * self.params.S0

        # Create models with shifted underlying
        params_up = HestonParameters(
            S0=self.params.S0 + shift, K=self.params.K, T=self.params.T, r=self.params.r, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_up = HestonModel(params_up)

        params_down = HestonParameters(
            S0=self.params.S0 - shift, K=self.params.K, T=self.params.T, r=self.params.r, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_down = HestonModel(params_down)

        return (model_up.delta('call') - model_down.delta('call')) / (2 * shift)

    def vega(self) -> float:
        """Calculate vega with respect to initial volatility"""
        vol_shift = 0.01  # 1% absolute shift in initial volatility

        # Shift initial variance up (volatility squared)
        v0_up = (np.sqrt(self.params.v0) + vol_shift) ** 2
        params_up = HestonParameters(
            S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r, q=self.params.q,
            v0=v0_up, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_up = HestonModel(params_up)

        # Shift initial variance down
        v0_down = max(0.0001, (np.sqrt(self.params.v0) - vol_shift) ** 2)
        params_down = HestonParameters(
            S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r, q=self.params.q,
            v0=v0_down, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_down = HestonModel(params_down)

        # Return vega as sensitivity to volatility (not variance)
        return (model_up.call_price() - model_down.call_price()) / (2 * vol_shift) / 100

    def theta(self, option_type: str = 'call') -> float:
        """Calculate theta using finite difference"""
        time_shift = min(1/365, self.params.T * 0.01)  # 1 day or 1% of remaining time

        # Shift time to expiry down
        params_shifted = HestonParameters(
            S0=self.params.S0, K=self.params.K, T=max(0.001, self.params.T - time_shift), r=self.params.r, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_shifted = HestonModel(params_shifted)

        current_price = self.option_price(option_type)
        shifted_price = model_shifted.option_price(option_type)

        return (shifted_price - current_price) / time_shift

    def rho(self, option_type: str = 'call') -> float:
        """Calculate rho using finite difference"""
        rate_shift = 0.0001  # 1 basis point

        # Shift rate up
        params_up = HestonParameters(
            S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r + rate_shift, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_up = HestonModel(params_up)

        # Shift rate down
        params_down = HestonParameters(
            S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r - rate_shift, q=self.params.q,
            v0=self.params.v0, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
        )
        model_down = HestonModel(params_down)

        return (model_up.option_price(option_type) - model_down.option_price(option_type)) / (2 * rate_shift) / 100

    def implied_volatility(self, market_price: float, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility by adjusting initial variance.
        Note: This is a simplified approach - full Heston IV surface calibration would be more complex.
        """
        # Initial guess for volatility (sqrt of initial variance)
        vol = np.sqrt(self.params.v0)

        for i in range(max_iterations):
            # Update initial variance
            temp_params = HestonParameters(
                S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r, q=self.params.q,
                v0=vol**2, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
            )
            temp_model = HestonModel(temp_params)

            try:
                model_price = temp_model.option_price(option_type)
                price_diff = model_price - market_price

                if abs(price_diff) < tolerance:
                    return vol

                # Calculate vega for Newton-Raphson update
                vega_value = temp_model.vega() * 100  # Convert to decimal

                if abs(vega_value) < 1e-10:
                    raise ValueError("Vega too small for implied volatility calculation")

                vol = vol - price_diff / vega_value
                vol = max(vol, 1e-6)  # Ensure positive volatility

            except:
                # If Heston pricing fails, fall back to bisection
                return self._implied_vol_bisection(market_price, option_type, max_iterations, tolerance)

        raise ValueError(f"Implied volatility did not converge after {max_iterations} iterations")

    def _implied_vol_bisection(self, market_price: float, option_type: str,
                              max_iterations: int, tolerance: float) -> float:
        """Fallback bisection method for implied volatility"""
        vol_low, vol_high = 0.001, 3.0

        for _ in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2

            temp_params = HestonParameters(
                S0=self.params.S0, K=self.params.K, T=self.params.T, r=self.params.r, q=self.params.q,
                v0=vol_mid**2, kappa=self.params.kappa, theta=self.params.theta, xi=self.params.xi, rho=self.params.rho
            )
            temp_model = HestonModel(temp_params)

            try:
                model_price = temp_model.option_price(option_type)
                price_diff = model_price - market_price

                if abs(price_diff) < tolerance:
                    return vol_mid

                if price_diff > 0:
                    vol_high = vol_mid
                else:
                    vol_low = vol_mid

            except:
                vol_high = vol_mid  # Assume pricing failed due to high volatility

        return (vol_low + vol_high) / 2

    def calibrate(self, market_prices: list, strikes: list, maturities: list,
                  option_types: list, initial_params: Optional[dict] = None) -> dict:
        """Calibrate Heston parameters to market prices"""

        if initial_params is None:
            initial_params = {
                'v0': 0.04,
                'kappa': 2.0,
                'theta': 0.04,
                'xi': 0.3,
                'rho': -0.5
            }

        def objective(params):
            v0, kappa, theta, xi, rho = params

            # Parameter constraints
            if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) >= 1:
                return 1e10

            # Feller condition
            if 2 * kappa * theta <= xi**2:
                return 1e10

            total_error = 0
            for i, (market_price, K, T, option_type) in enumerate(
                zip(market_prices, strikes, maturities, option_types)
            ):
                temp_params = HestonParameters(
                    S0=self.params.S0, K=K, T=T, r=self.params.r, q=self.params.q,
                    v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho
                )
                temp_model = HestonModel(temp_params)

                try:
                    model_price = temp_model.option_price(option_type)
                    error = (model_price - market_price)**2
                    total_error += error
                except:
                    total_error += 1e6  # Penalty for failed pricing

            return total_error

        # Initial guess
        x0 = [initial_params['v0'], initial_params['kappa'],
              initial_params['theta'], initial_params['xi'], initial_params['rho']]

        # Bounds
        bounds = [(1e-6, 1), (1e-6, 10), (1e-6, 1), (1e-6, 2), (-0.99, 0.99)]

        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        if result.success:
            v0, kappa, theta, xi, rho = result.x
            return {
                'v0': v0,
                'kappa': kappa,
                'theta': theta,
                'xi': xi,
                'rho': rho,
                'success': True,
                'rmse': np.sqrt(result.fun / len(market_prices))
            }
        else:
            return {'success': False, 'message': result.message}


def price_heston_option(S0: float, K: float, T: float, r: float, v0: float,
                       kappa: float, theta: float, xi: float, rho: float,
                       option_type: str = 'call', q: float = 0.0, method: str = 'analytic') -> float:
    """Convenience function to price options under Heston model"""
    params = HestonParameters(S0=S0, K=K, T=T, r=r, v0=v0, kappa=kappa,
                             theta=theta, xi=xi, rho=rho, q=q)
    model = HestonModel(params)

    if method == 'analytic':
        return model.option_price(option_type)
    elif method == 'mc':
        return model.monte_carlo_price(option_type=option_type)
    else:
        raise ValueError("Method must be 'analytic' or 'mc'")


if __name__ == "__main__":
    # Example usage
    params = HestonParameters(
        S0=100, K=105, T=0.25, r=0.05, v0=0.04,
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, q=0.02
    )

    model = HestonModel(params)

    print("Heston Model Results:")
    print(f"Call Price (Analytic): {model.call_price():.4f}")
    print(f"Put Price (Analytic): {model.put_price():.4f}")
    print(f"Call Price (Monte Carlo): {model.monte_carlo_price(option_type='call', seed=42):.4f}")
    print(f"Put Price (Monte Carlo): {model.monte_carlo_price(option_type='put', seed=42):.4f}")

    print(f"\nGreeks:")
    print(f"Delta (Call): {model.delta('call'):.4f}")
    print(f"Gamma: {model.gamma():.4f}")
    print(f"Vega: {model.vega():.4f}")
    print(f"Theta (Call): {model.theta('call'):.4f}")
    print(f"Rho (Call): {model.rho('call'):.4f}")

    print(f"\nGreeks Dictionary:")
    print(model.greeks('call'))


# Register with the model factory
from .base_model import ModelFactory
ModelFactory.register_model('heston', HestonModel)
ModelFactory.register_model('stochastic-vol', HestonModel)