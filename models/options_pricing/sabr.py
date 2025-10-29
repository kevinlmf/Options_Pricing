"""
SABR (Stochastic Alpha Beta Rho) Model
Implements the SABR stochastic volatility model for option pricing

dF_t = σ_t F_t^β dW_1
dσ_t = α σ_t dW_2
dW_1 dW_2 = ρ dt

where:
- F: Forward price
- σ: Stochastic volatility
- α: Volatility of volatility (volvol)
- β: CEV parameter (0 for normal, 1 for lognormal)
- ρ: Correlation between price and volatility
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Tuple, Optional

# Try to import C++ accelerated module
try:
    import sabr_cpp
    SABR_CPP_AVAILABLE = True
except ImportError:
    SABR_CPP_AVAILABLE = False
class SABRModel:
    """
    SABR Model for Options Pricing

    Hagan et al. (2002) formula for implied volatility approximation
    """

    def __init__(self, F: float, K: float, T: float, r: float,
                 alpha: float, beta: float, rho: float, nu: float):
        """
        Initialize SABR model parameters

        Parameters:
        -----------
        F : float
            Forward price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate
        alpha : float
            Initial volatility level
        beta : float
            CEV parameter (0 <= beta <= 1)
            beta=0: Normal (Bachelier)
            beta=0.5: CIR/Square-root process
            beta=1: Lognormal (Black-Scholes)
        rho : float
            Correlation between price and volatility (-1 <= rho <= 1)
        nu : float
            Volatility of volatility (volvol)
        """
        super().__init__()
        self.F = F
        self.K = K
        self.T = T
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate SABR parameters"""
        assert 0 <= self.beta <= 1, "Beta must be between 0 and 1"
        assert -1 <= self.rho <= 1, "Rho must be between -1 and 1"
        assert self.alpha > 0, "Alpha must be positive"
        assert self.nu >= 0, "Nu must be non-negative"
        assert self.T > 0, "Time to expiration must be positive"

    def implied_volatility_hagan(self, use_cpp: bool = True) -> float:
        """
        Calculate implied volatility using Hagan's approximation formula

        Parameters:
        -----------
        use_cpp : bool
            Use C++ acceleration if available (default: True)

        Returns:
        --------
        float : Implied volatility
        """
        # Try C++ acceleration first if available and requested
        if use_cpp and SABR_CPP_AVAILABLE:
            try:
                result = sabr_cpp.price_option(
                    F=self.F, K=self.K, T=self.T, r=self.r,
                    alpha=self.alpha, beta=self.beta, rho=self.rho, nu=self.nu,
                    option_type='call'
                )
                return result['implied_vol']
            except Exception:
                pass  # Fallback to Python implementation

        F, K, T = self.F, self.K, self.T
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        # Handle ATM case separately
        if abs(F - K) < 1e-10:
            return self._atm_implied_volatility()

        # Calculate intermediate values
        FK = F * K
        log_FK = np.log(F / K)

        # z parameter
        z = (nu / alpha) * np.power(FK, (1 - beta) / 2) * log_FK

        # x(z) function - use approximation to avoid division by zero
        if abs(z) < 1e-6:
            x_z = 1.0
        else:
            numerator = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            x_z = z / numerator

        # Calculate implied volatility
        # Term 1: Main term
        term1_numerator = alpha
        term1_denominator = np.power(FK, (1 - beta) / 2) * (
            1 + ((1 - beta)**2 / 24) * log_FK**2 +
            ((1 - beta)**4 / 1920) * log_FK**4
        )
        term1 = term1_numerator / term1_denominator

        # Term 2: Time-dependent correction
        term2 = 1 + T * (
            ((1 - beta)**2 / 24) * (alpha**2 / np.power(FK, 1 - beta)) +
            (rho * beta * nu * alpha / (4 * np.power(FK, (1 - beta) / 2))) +
            ((2 - 3 * rho**2) / 24) * nu**2
        )

        implied_vol = term1 * x_z * term2

        return max(implied_vol, 1e-6)  # Ensure positive volatility

    def _atm_implied_volatility(self) -> float:
        """
        Calculate ATM implied volatility (F ≈ K)

        Returns:
        --------
        float : ATM implied volatility
        """
        F = self.F
        alpha, beta, rho, nu, T = self.alpha, self.beta, self.rho, self.nu, self.T

        # ATM formula
        term1 = alpha / np.power(F, 1 - beta)
        term2 = 1 + T * (
            ((1 - beta)**2 / 24) * (alpha**2 / np.power(F, 2 * (1 - beta))) +
            (rho * beta * nu * alpha / (4 * np.power(F, 1 - beta))) +
            ((2 - 3 * rho**2) / 24) * nu**2
        )

        return term1 * term2

    def price(self, option_type: str = 'call', use_cpp: bool = True) -> float:
        """
        Calculate option price using SABR implied volatility

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        use_cpp : bool
            Use C++ acceleration if available (default: True)

        Returns:
        --------
        float : Option price
        """
        # Try C++ acceleration first if available and requested
        if use_cpp and SABR_CPP_AVAILABLE:
            try:
                result = sabr_cpp.price_option(
                    F=self.F, K=self.K, T=self.T, r=self.r,
                    alpha=self.alpha, beta=self.beta, rho=self.rho, nu=self.nu,
                    option_type=option_type
                )
                return result['price']
            except Exception:
                pass  # Fallback to Python implementation

        # Get implied volatility from SABR
        implied_vol = self.implied_volatility_hagan(use_cpp=False)

        # Use Black-Scholes formula with SABR implied vol
        S = self.F * np.exp(-self.r * self.T)  # Convert forward to spot
        return self._black_scholes_price(S, self.K, self.T, self.r,
                                         implied_vol, option_type)

    def _black_scholes_price(self, S: float, K: float, T: float, r: float,
                            sigma: float, option_type: str) -> float:
        """Black-Scholes pricing formula"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def greeks(self, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks using finite difference

        Returns:
        --------
        dict : Dictionary of Greeks (delta, gamma, vega, theta, rho)
        """
        epsilon = 0.01
        dt = 1.0 / 365.0

        # Delta
        sabr_up = SABRModel(self.F * (1 + epsilon), self.K, self.T, self.r,
                           self.alpha, self.beta, self.rho, self.nu)
        sabr_down = SABRModel(self.F * (1 - epsilon), self.K, self.T, self.r,
                             self.alpha, self.beta, self.rho, self.nu)
        delta = (sabr_up.price(option_type) - sabr_down.price(option_type)) / (
            2 * epsilon * self.F)

        # Gamma
        gamma = (sabr_up.price(option_type) - 2 * self.price(option_type) +
                sabr_down.price(option_type)) / ((epsilon * self.F)**2)

        # Vega (with respect to alpha)
        sabr_alpha_up = SABRModel(self.F, self.K, self.T, self.r,
                                 self.alpha * (1 + epsilon), self.beta,
                                 self.rho, self.nu)
        vega = (sabr_alpha_up.price(option_type) - self.price(option_type)) / (
            epsilon * self.alpha)

        # Theta
        if self.T > dt:
            sabr_theta = SABRModel(self.F, self.K, self.T - dt, self.r,
                                  self.alpha, self.beta, self.rho, self.nu)
            theta = sabr_theta.price(option_type) - self.price(option_type)
        else:
            theta = 0.0

        # Rho (interest rate sensitivity)
        sabr_rho = SABRModel(self.F, self.K, self.T, self.r + epsilon,
                            self.alpha, self.beta, self.rho, self.nu)
        rho_greek = (sabr_rho.price(option_type) - self.price(option_type)) / epsilon

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho_greek
        }

    def calibrate(self, market_prices: np.ndarray, strikes: np.ndarray,
                 maturities: np.ndarray, initial_guess: Optional[Dict] = None) -> Dict:
        """
        Calibrate SABR parameters to market prices

        Parameters:
        -----------
        market_prices : np.ndarray
            Market option prices
        strikes : np.ndarray
            Option strikes
        maturities : np.ndarray
            Option maturities
        initial_guess : dict, optional
            Initial parameter guesses

        Returns:
        --------
        dict : Calibrated parameters
        """
        if initial_guess is None:
            initial_guess = {
                'alpha': 0.3,
                'beta': 0.5,
                'rho': -0.3,
                'nu': 0.4
            }

        def objective(params):
            alpha, beta, rho, nu = params

            # Ensure valid parameters
            beta = np.clip(beta, 0.01, 0.99)
            rho = np.clip(rho, -0.99, 0.99)
            alpha = max(alpha, 0.01)
            nu = max(nu, 0.01)

            model_prices = []
            for i, (K, T) in enumerate(zip(strikes, maturities)):
                try:
                    sabr = SABRModel(self.F, K, T, self.r, alpha, beta, rho, nu)
                    model_prices.append(sabr.price())
                except:
                    model_prices.append(1e10)  # Penalize invalid parameters

            model_prices = np.array(model_prices)
            return np.sum((model_prices - market_prices)**2)

        # Optimization bounds
        bounds = [
            (0.01, 2.0),    # alpha
            (0.01, 0.99),   # beta
            (-0.99, 0.99),  # rho
            (0.01, 2.0)     # nu
        ]

        x0 = [initial_guess['alpha'], initial_guess['beta'],
              initial_guess['rho'], initial_guess['nu']]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        return {
            'alpha': result.x[0],
            'beta': result.x[1],
            'rho': result.x[2],
            'nu': result.x[3],
            'success': result.success,
            'rmse': np.sqrt(result.fun / len(market_prices))
        }

    def simulate_paths(self, n_paths: int = 10000, n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo simulation of SABR model paths

        Parameters:
        -----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps

        Returns:
        --------
        tuple : (price_paths, volatility_paths)
        """
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize arrays
        F_paths = np.zeros((n_steps + 1, n_paths))
        sigma_paths = np.zeros((n_steps + 1, n_paths))

        F_paths[0, :] = self.F
        sigma_paths[0, :] = self.alpha

        # Generate correlated Brownian motions
        for i in range(n_steps):
            # Independent normal random variables
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)

            # Correlate them
            dW1 = z1 * sqrt_dt
            dW2 = (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2) * sqrt_dt

            # Update volatility
            sigma_paths[i + 1, :] = sigma_paths[i, :] + (
                self.nu * sigma_paths[i, :] * dW2
            )
            sigma_paths[i + 1, :] = np.maximum(sigma_paths[i + 1, :], 1e-6)

            # Update forward price
            F_paths[i + 1, :] = F_paths[i, :] + (
                sigma_paths[i, :] * np.power(F_paths[i, :], self.beta) * dW1
            )
            F_paths[i + 1, :] = np.maximum(F_paths[i + 1, :], 1e-6)

        return F_paths, sigma_paths


def sabr_implied_volatility(F: float, K: float, T: float, alpha: float,
                            beta: float, rho: float, nu: float) -> float:
    """
    Standalone function to compute SABR implied volatility

    Parameters:
    -----------
    F : float
        Forward price
    K : float
        Strike price
    T : float
        Time to expiration
    alpha : float
        Initial volatility
    beta : float
        CEV parameter
    rho : float
        Correlation
    nu : float
        Volatility of volatility

    Returns:
    --------
    float : Implied volatility
    """
    sabr = SABRModel(F, K, T, 0.0, alpha, beta, rho, nu)
    return sabr.implied_volatility_hagan()
