"""
Local Volatility Model (Dupire Model)
Implements local volatility surface for option pricing

The local volatility function σ_LV(K,T) satisfies Dupire's formula:
σ_LV²(K,T) = 2 * ∂C/∂T / (K² * ∂²C/∂K²)

where C(K,T) is the market call option price
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp2d
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Tuple, Optional, Callable
class LocalVolatilityModel:
    """
    Local Volatility Model for Options Pricing

    Implements Dupire's local volatility model with finite difference PDE solver
    """

    def __init__(self, S: float, r: float, q: float = 0.0):
        """
        Initialize Local Volatility model

        Parameters:
        -----------
        S : float
            Current spot price
        r : float
            Risk-free rate
        q : float
            Dividend yield (default 0)
        """
        super().__init__()
        self.S = S
        self.r = r
        self.q = q
        self.local_vol_surface = None
        self.strike_grid = None
        self.maturity_grid = None

    def set_local_vol_surface(self, strikes: np.ndarray, maturities: np.ndarray,
                             local_vols: np.ndarray):
        """
        Set the local volatility surface

        Parameters:
        -----------
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities
        local_vols : np.ndarray
            2D array of local volatilities (len(strikes) x len(maturities))
        """
        self.strike_grid = strikes
        self.maturity_grid = maturities

        # Create interpolation function for local volatility surface
        self.local_vol_surface = RectBivariateSpline(
            strikes, maturities, local_vols, kx=1, ky=1
        )

    def compute_local_vol_from_implied(self, strikes: np.ndarray,
                                       maturities: np.ndarray,
                                       implied_vols: np.ndarray) -> np.ndarray:
        """
        Compute local volatility surface from implied volatility surface
        using Dupire's formula

        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        maturities : np.ndarray
            Time to maturity
        implied_vols : np.ndarray
            Implied volatility surface (len(strikes) x len(maturities))

        Returns:
        --------
        np.ndarray : Local volatility surface
        """
        local_vols = np.zeros_like(implied_vols)

        # Create interpolation for implied volatility
        # Dynamically set spline degree based on data points
        kx = min(3, len(strikes) - 1)
        ky = min(3, len(maturities) - 1)
        kx = max(1, kx)  # At least linear
        ky = max(1, ky)

        iv_interp = RectBivariateSpline(strikes, maturities, implied_vols,
                                        kx=kx, ky=ky)

        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                try:
                    local_vols[i, j] = self._dupire_formula(K, T, iv_interp)
                except:
                    # Fallback to implied vol if calculation fails
                    local_vols[i, j] = implied_vols[i, j]

        return local_vols

    def _dupire_formula(self, K: float, T: float,
                       iv_surface: RectBivariateSpline) -> float:
        """
        Apply Dupire's formula to compute local volatility

        σ_LV²(K,T) = (∂C/∂T + qC + (r-q)K∂C/∂K) / (0.5K²∂²C/∂K²)

        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to maturity
        iv_surface : RectBivariateSpline
            Implied volatility interpolation function

        Returns:
        --------
        float : Local volatility
        """
        epsilon_K = 0.01 * K
        epsilon_T = 0.01 * T

        # Get implied volatility and compute call prices
        iv = float(iv_surface(K, T))
        C = self._bs_call_price(self.S, K, T, self.r, self.q, iv)

        # Finite difference for ∂C/∂T
        if T > epsilon_T:
            iv_T_up = float(iv_surface(K, T + epsilon_T))
            C_T_up = self._bs_call_price(self.S, K, T + epsilon_T,
                                        self.r, self.q, iv_T_up)
            dC_dT = (C_T_up - C) / epsilon_T
        else:
            dC_dT = 0

        # Finite difference for ∂C/∂K
        iv_K_up = float(iv_surface(K + epsilon_K, T))
        iv_K_down = float(iv_surface(K - epsilon_K, T))
        C_K_up = self._bs_call_price(self.S, K + epsilon_K, T,
                                     self.r, self.q, iv_K_up)
        C_K_down = self._bs_call_price(self.S, K - epsilon_K, T,
                                       self.r, self.q, iv_K_down)
        dC_dK = (C_K_up - C_K_down) / (2 * epsilon_K)

        # Finite difference for ∂²C/∂K²
        d2C_dK2 = (C_K_up - 2 * C + C_K_down) / (epsilon_K**2)

        # Avoid division by zero
        if abs(d2C_dK2) < 1e-10:
            return iv

        # Dupire's formula
        numerator = dC_dT + self.q * C + (self.r - self.q) * K * dC_dK
        denominator = 0.5 * K**2 * d2C_dK2

        local_var = numerator / denominator

        # Ensure non-negative and reasonable values
        local_var = max(local_var, 0.01)
        local_vol = np.sqrt(local_var)

        return min(local_vol, 5.0)  # Cap at 500% vol

    def _bs_call_price(self, S: float, K: float, T: float, r: float,
                      q: float, sigma: float) -> float:
        """Black-Scholes call option price"""
        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call

    def price(self, K: float, T: float, option_type: str = 'call',
             n_space: int = 200, n_time: int = 100) -> float:
        """
        Price option using finite difference PDE solver with local volatility

        Parameters:
        -----------
        K : float
            Strike price
        T : float
            Time to maturity
        option_type : str
            'call' or 'put'
        n_space : int
            Number of spatial grid points
        n_time : int
            Number of time steps

        Returns:
        --------
        float : Option price
        """
        if self.local_vol_surface is None:
            raise ValueError("Local volatility surface not set. Call set_local_vol_surface first.")

        # Set up spatial grid
        S_max = 3 * self.S
        S_min = 0.1 * self.S
        S_grid = np.linspace(S_min, S_max, n_space)
        dS = S_grid[1] - S_grid[0]

        # Set up time grid
        dt = T / n_time

        # Initialize option value at maturity
        if option_type.lower() == 'call':
            V = np.maximum(S_grid - K, 0)
        else:
            V = np.maximum(K - S_grid, 0)

        # Finite difference coefficients (implicit scheme)
        A = np.zeros((n_space, n_space))

        for t_idx in range(n_time):
            t = T - t_idx * dt

            # Build coefficient matrix
            for i in range(1, n_space - 1):
                S = S_grid[i]
                sigma = float(self.local_vol_surface(S, t))

                # Coefficients for implicit scheme
                alpha = 0.5 * dt * (sigma**2 * S**2 / dS**2 - (self.r - self.q) * S / dS)
                beta = -dt * (sigma**2 * S**2 / dS**2 + self.r)
                gamma = 0.5 * dt * (sigma**2 * S**2 / dS**2 + (self.r - self.q) * S / dS)

                A[i, i-1] = -alpha
                A[i, i] = 1 - beta
                A[i, i+1] = -gamma

            # Boundary conditions
            A[0, 0] = 1
            A[-1, -1] = 1

            # Apply boundary conditions to RHS
            if option_type.lower() == 'call':
                V[0] = 0
                V[-1] = S_max - K * np.exp(-self.r * (t - dt))
            else:
                V[0] = K * np.exp(-self.r * (t - dt))
                V[-1] = 0

            # Solve linear system
            V = np.linalg.solve(A, V)

        # Interpolate to get price at S_0
        return np.interp(self.S, S_grid, V)

    def greeks(self, K: float, T: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks

        Returns:
        --------
        dict : Dictionary of Greeks
        """
        epsilon_S = 0.01 * self.S
        epsilon_T = 1.0 / 365.0
        epsilon_r = 0.0001

        # Delta
        S_up = self.S
        self.S = self.S + epsilon_S
        price_up = self.price(K, T, option_type)
        self.S = S_up - epsilon_S
        price_down = self.price(K, T, option_type)
        self.S = S_up

        delta = (price_up - price_down) / (2 * epsilon_S)

        # Gamma
        price_0 = self.price(K, T, option_type)
        gamma = (price_up - 2 * price_0 + price_down) / (epsilon_S**2)

        # Theta
        if T > epsilon_T:
            theta = self.price(K, T - epsilon_T, option_type) - price_0
        else:
            theta = 0.0

        # Vega (sensitivity to volatility level - approximate)
        # For local vol, vega is more complex as it depends on the entire surface
        vega = 0.0  # Would require sensitivity to local vol surface

        # Rho
        r_orig = self.r
        self.r = r_orig + epsilon_r
        price_r_up = self.price(K, T, option_type)
        self.r = r_orig
        rho = (price_r_up - price_0) / epsilon_r

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    def simulate_paths(self, T: float, n_paths: int = 10000,
                      n_steps: int = 100) -> np.ndarray:
        """
        Monte Carlo simulation with local volatility

        Parameters:
        -----------
        T : float
            Time horizon
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps

        Returns:
        --------
        np.ndarray : Simulated price paths
        """
        if self.local_vol_surface is None:
            raise ValueError("Local volatility surface not set.")

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        S_paths = np.zeros((n_steps + 1, n_paths))
        S_paths[0, :] = self.S

        for i in range(n_steps):
            t = i * dt
            dW = np.random.standard_normal(n_paths) * sqrt_dt

            # Get local volatility for each path
            local_vols = np.array([
                float(self.local_vol_surface(S, t))
                for S in S_paths[i, :]
            ])

            # Update prices
            S_paths[i + 1, :] = S_paths[i, :] * np.exp(
                (self.r - self.q - 0.5 * local_vols**2) * dt +
                local_vols * dW
            )

        return S_paths

    def calibrate_to_market(self, strikes: np.ndarray, maturities: np.ndarray,
                           market_prices: np.ndarray,
                           option_types: np.ndarray) -> Dict:
        """
        Calibrate local volatility surface to market prices

        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        maturities : np.ndarray
            Maturities
        market_prices : np.ndarray
            Market option prices
        option_types : np.ndarray
            Option types ('call' or 'put')

        Returns:
        --------
        dict : Calibration results
        """
        # First, back out implied volatilities
        implied_vols = np.zeros((len(strikes), len(np.unique(maturities))))

        unique_maturities = np.unique(maturities)

        for i, K in enumerate(strikes):
            for j, T in enumerate(unique_maturities):
                mask = (strikes == K) & (maturities == T)
                if np.any(mask):
                    market_price = market_prices[mask][0]
                    option_type = option_types[mask][0]

                    # Back out implied vol using Black-Scholes
                    iv = self._implied_volatility_newton(
                        market_price, K, T, option_type
                    )
                    implied_vols[i, j] = iv

        # Compute local volatility surface
        local_vols = self.compute_local_vol_from_implied(
            strikes, unique_maturities, implied_vols
        )

        # Set the local volatility surface
        self.set_local_vol_surface(strikes, unique_maturities, local_vols)

        return {
            'implied_vols': implied_vols,
            'local_vols': local_vols,
            'strikes': strikes,
            'maturities': unique_maturities
        }

    def _implied_volatility_newton(self, market_price: float, K: float,
                                   T: float, option_type: str,
                                   max_iter: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        sigma = 0.3  # Initial guess

        for _ in range(max_iter):
            price = self._bs_call_price(self.S, K, T, self.r, self.q, sigma)

            if option_type.lower() == 'put':
                # Put-call parity
                price = price - self.S * np.exp(-self.q * T) + K * np.exp(-self.r * T)

            diff = price - market_price

            if abs(diff) < 1e-6:
                break

            # Vega (derivative of price w.r.t. sigma)
            d1 = (np.log(self.S / K) + (self.r - self.q + 0.5 * sigma**2) * T) / (
                sigma * np.sqrt(T))
            vega = self.S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T)

            if vega < 1e-10:
                break

            sigma = sigma - diff / vega
            sigma = max(sigma, 0.01)  # Ensure positive

        return sigma
