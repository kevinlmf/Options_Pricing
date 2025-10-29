"""
Sample Data Generators

Implementation of synthetic data generators for testing and research.
Follows the mathematical framework: Pure Math → Applied Math → Data Generation

Mathematical Foundation:
- Pure Math: Random processes, distribution theory, correlation structures
- Applied Math: Stochastic models, Monte Carlo simulation, factor models
- Data Generation: Realistic financial data simulation for testing and backtesting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any
from scipy import linalg
from scipy.stats import multivariate_normal, t as student_t
from datetime import datetime, timedelta


class SampleDataGenerator:
    """
    Base class for generating synthetic financial data.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize data generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_gbm_paths(self, S0: float, mu: float, sigma: float, T: float,
                          n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Generate Geometric Brownian Motion paths.

        Args:
            S0: Initial price
            mu: Drift rate
            sigma: Volatility
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths to generate

        Returns:
            Array of shape (n_paths, n_steps+1) with price paths
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Generate random shocks
        dW = np.random.normal(0, sqrt_dt, (n_paths, n_steps))

        # Initialize price paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        # Generate paths using exact solution
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i])

        return paths

    def generate_jump_diffusion_paths(self, S0: float, mu: float, sigma: float,
                                     lambda_jump: float, mu_jump: float, sigma_jump: float,
                                     T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Generate Jump Diffusion (Merton) model paths.

        Args:
            S0: Initial price
            mu: Drift rate
            sigma: Diffusion volatility
            lambda_jump: Jump intensity
            mu_jump: Jump size mean
            sigma_jump: Jump size volatility
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths

        Returns:
            Array of price paths with jumps
        """
        dt = T / n_steps

        # Generate base GBM paths
        gbm_paths = self.generate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths)

        # Add jumps
        paths = gbm_paths.copy()
        for path_idx in range(n_paths):
            for step in range(1, n_steps + 1):
                # Check if jump occurs
                if np.random.poisson(lambda_jump * dt) > 0:
                    # Generate jump size
                    jump_size = np.random.normal(mu_jump, sigma_jump)
                    paths[path_idx, step] *= np.exp(jump_size)

        return paths

    def generate_heston_paths(self, S0: float, v0: float, kappa: float, theta: float,
                             xi: float, rho: float, r: float, T: float,
                             n_steps: int, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Heston stochastic volatility paths using Euler discretization.

        Args:
            S0: Initial stock price
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-run variance
            xi: Vol of vol
            rho: Correlation between price and vol
            r: Risk-free rate
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths

        Returns:
            Tuple of (price_paths, variance_paths)
        """
        dt = T / n_steps

        # Initialize paths
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))

        S_paths[:, 0] = S0
        v_paths[:, 0] = v0

        # Generate correlated random numbers
        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)

            # Update variance (ensure non-negative)
            v_paths[:, i + 1] = np.maximum(
                v_paths[:, i] + kappa * (theta - v_paths[:, i]) * dt + xi * np.sqrt(v_paths[:, i]) * np.sqrt(dt) * Z2,
                0.0001  # Floor variance at small positive value
            )

            # Update stock price
            S_paths[:, i + 1] = S_paths[:, i] * np.exp(
                (r - 0.5 * v_paths[:, i]) * dt + np.sqrt(v_paths[:, i]) * np.sqrt(dt) * Z1
            )

        return S_paths, v_paths

    def generate_price_series(self, n_periods: int, initial_price: float = 100.0,
                             drift: float = 0.05, volatility: float = 0.2) -> np.ndarray:
        """
        Generate a simple price series using Geometric Brownian Motion.

        This is a convenience method that wraps generate_gbm_paths for single path generation.

        Args:
            n_periods: Number of time periods
            initial_price: Starting price
            drift: Annual drift rate
            volatility: Annual volatility

        Returns:
            Array of prices
        """
        # Assume daily data, so T = n_periods / 252
        T = n_periods / 252.0

        # Generate single path
        price_path = self.generate_gbm_paths(
            S0=initial_price,
            mu=drift,
            sigma=volatility,
            T=T,
            n_steps=n_periods - 1,  # n_steps is one less than n_periods
            n_paths=1
        )

        # Return flattened single path
        return price_path.flatten()


class CorrelatedReturnsGenerator(SampleDataGenerator):
    """
    Generator for correlated multi-asset returns.
    """

    def __init__(self, random_seed: Optional[int] = None):
        super().__init__(random_seed)

    def generate_multivariate_normal_returns(self, mean_returns: np.ndarray,
                                            cov_matrix: np.ndarray,
                                            n_periods: int) -> np.ndarray:
        """
        Generate multivariate normal returns.

        Args:
            mean_returns: Mean return vector
            cov_matrix: Covariance matrix
            n_periods: Number of periods

        Returns:
            Returns matrix (n_periods, n_assets)
        """
        returns = multivariate_normal.rvs(mean=mean_returns, cov=cov_matrix, size=n_periods)
        return returns.reshape(n_periods, -1)

    def generate_factor_model_returns(self, factor_loadings: np.ndarray,
                                    factor_returns: np.ndarray,
                                    idiosyncratic_vol: np.ndarray) -> np.ndarray:
        """
        Generate returns using a factor model.
        R_i = β_i * F + ε_i

        Args:
            factor_loadings: Beta matrix (n_assets, n_factors)
            factor_returns: Factor return matrix (n_periods, n_factors)
            idiosyncratic_vol: Idiosyncratic volatility vector

        Returns:
            Asset returns matrix (n_periods, n_assets)
        """
        n_periods, n_factors = factor_returns.shape
        n_assets = factor_loadings.shape[0]

        # Systematic component
        systematic_returns = factor_returns @ factor_loadings.T

        # Idiosyncratic component
        idiosyncratic_returns = np.random.normal(
            0, idiosyncratic_vol, (n_periods, n_assets)
        )

        return systematic_returns + idiosyncratic_returns

    def generate_garch_returns(self, mu: float, omega: float, alpha: float,
                              beta: float, n_periods: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate GARCH(1,1) return series.
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

        Args:
            mu: Mean return
            omega: GARCH intercept
            alpha: ARCH parameter
            beta: GARCH parameter
            n_periods: Number of periods

        Returns:
            Tuple of (returns, conditional_volatility)
        """
        returns = np.zeros(n_periods)
        vol = np.zeros(n_periods)

        # Initial values
        vol[0] = np.sqrt(omega / (1 - alpha - beta))  # Unconditional volatility
        returns[0] = mu + vol[0] * np.random.standard_normal()

        for t in range(1, n_periods):
            # Update volatility
            vol[t] = np.sqrt(omega + alpha * (returns[t-1] - mu)**2 + beta * vol[t-1]**2)

            # Generate return
            returns[t] = mu + vol[t] * np.random.standard_normal()

        return returns, vol

    def generate_regime_switching_returns(self, regime_params: List[Dict],
                                         transition_matrix: np.ndarray,
                                         n_periods: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate returns with regime switching.

        Args:
            regime_params: List of regime parameters [{'mu': float, 'sigma': float}, ...]
            transition_matrix: Regime transition probability matrix
            n_periods: Number of periods

        Returns:
            Tuple of (returns, regime_states)
        """
        n_regimes = len(regime_params)
        returns = np.zeros(n_periods)
        regimes = np.zeros(n_periods, dtype=int)

        # Start in regime 0
        current_regime = 0
        regimes[0] = current_regime

        for t in range(n_periods):
            # Generate return for current regime
            mu = regime_params[current_regime]['mu']
            sigma = regime_params[current_regime]['sigma']
            returns[t] = np.random.normal(mu, sigma)

            # Update regime for next period
            if t < n_periods - 1:
                transition_probs = transition_matrix[current_regime, :]
                current_regime = np.random.choice(n_regimes, p=transition_probs)
                regimes[t + 1] = current_regime

        return returns, regimes

    def generate_copula_returns(self, marginal_distributions: List[str],
                              marginal_params: List[Dict],
                              correlation_matrix: np.ndarray,
                              n_periods: int) -> np.ndarray:
        """
        Generate returns using copula approach.

        Args:
            marginal_distributions: List of distribution names
            marginal_params: List of parameter dictionaries for each marginal
            correlation_matrix: Gaussian copula correlation matrix
            n_periods: Number of periods

        Returns:
            Returns matrix with specified marginal distributions and correlation
        """
        n_assets = len(marginal_distributions)

        # Generate multivariate normal variables
        mvn_samples = multivariate_normal.rvs(
            mean=np.zeros(n_assets),
            cov=correlation_matrix,
            size=n_periods
        )

        # Convert to uniform variables via normal CDF
        uniform_samples = stats.norm.cdf(mvn_samples)

        # Transform to desired marginal distributions
        returns = np.zeros((n_periods, n_assets))

        for i in range(n_assets):
            dist_name = marginal_distributions[i]
            params = marginal_params[i]

            if dist_name == 'normal':
                returns[:, i] = stats.norm.ppf(uniform_samples[:, i],
                                             loc=params.get('mu', 0),
                                             scale=params.get('sigma', 1))
            elif dist_name == 't':
                returns[:, i] = stats.t.ppf(uniform_samples[:, i],
                                          df=params.get('df', 5),
                                          loc=params.get('mu', 0),
                                          scale=params.get('sigma', 1))
            elif dist_name == 'skewnorm':
                returns[:, i] = stats.skewnorm.ppf(uniform_samples[:, i],
                                                 a=params.get('skew', 0),
                                                 loc=params.get('mu', 0),
                                                 scale=params.get('sigma', 1))

        return returns


class VolatilityModelGenerator(SampleDataGenerator):
    """
    Generator for various volatility models and processes.
    """

    def generate_stochastic_volatility_paths(self, v0: float, kappa: float, theta: float,
                                           sigma_v: float, T: float, n_steps: int,
                                           n_paths: int = 1, model: str = 'cir') -> np.ndarray:
        """
        Generate stochastic volatility paths.

        Args:
            v0: Initial volatility
            kappa: Mean reversion speed
            theta: Long-run volatility
            sigma_v: Vol of vol
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            model: Model type ('cir', 'ornstein_uhlenbeck')

        Returns:
            Volatility paths
        """
        dt = T / n_steps
        vol_paths = np.zeros((n_paths, n_steps + 1))
        vol_paths[:, 0] = v0

        for i in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)

            if model == 'cir':  # Cox-Ingersoll-Ross
                vol_paths[:, i + 1] = np.maximum(
                    vol_paths[:, i] + kappa * (theta - vol_paths[:, i]) * dt +
                    sigma_v * np.sqrt(vol_paths[:, i]) * dW,
                    0.001  # Ensure non-negative
                )
            elif model == 'ornstein_uhlenbeck':
                vol_paths[:, i + 1] = (vol_paths[:, i] + kappa * (theta - vol_paths[:, i]) * dt +
                                      sigma_v * dW)

        return vol_paths

    def generate_realized_volatility(self, returns: np.ndarray, frequency: str = 'daily',
                                   estimation_window: int = 30) -> np.ndarray:
        """
        Generate realized volatility estimates from high-frequency-like returns.

        Args:
            returns: High frequency returns
            frequency: Target frequency for RV ('daily', 'weekly')
            estimation_window: Window for volatility estimation

        Returns:
            Realized volatility series
        """
        if frequency == 'daily':
            # Sum of squared returns within each day
            rv = np.sqrt(np.sum(returns**2, axis=0)) if returns.ndim > 1 else np.sqrt(np.sum(returns**2))
        else:
            # Rolling realized volatility
            rv = pd.Series(returns).rolling(window=estimation_window).apply(
                lambda x: np.sqrt(np.sum(x**2))
            ).values

        return rv

    def generate_implied_volatility_surface(self, S0: float, r: float, T_range: np.ndarray,
                                          K_range: np.ndarray, vol_params: Dict) -> np.ndarray:
        """
        Generate synthetic implied volatility surface.

        Args:
            S0: Current stock price
            r: Risk-free rate
            T_range: Time to expiration range
            K_range: Strike price range
            vol_params: Volatility surface parameters

        Returns:
            Implied volatility surface (strikes x maturities)
        """
        n_strikes = len(K_range)
        n_maturities = len(T_range)
        iv_surface = np.zeros((n_strikes, n_maturities))

        # Base volatility
        base_vol = vol_params.get('base_vol', 0.20)
        vol_vol = vol_params.get('vol_vol', 0.05)
        skew_param = vol_params.get('skew', -0.1)
        term_structure = vol_params.get('term_structure', 0.02)

        for i, K in enumerate(K_range):
            for j, T in enumerate(T_range):
                # Moneyness effect (volatility smile/skew)
                moneyness = np.log(K / S0)
                moneyness_effect = skew_param * moneyness + 0.5 * vol_vol * moneyness**2

                # Term structure effect
                term_effect = term_structure * np.sqrt(T)

                # Final IV
                iv_surface[i, j] = base_vol + moneyness_effect + term_effect

                # Add some noise
                iv_surface[i, j] += np.random.normal(0, 0.01)

                # Ensure positive
                iv_surface[i, j] = max(iv_surface[i, j], 0.05)

        return iv_surface


if __name__ == "__main__":
    # Demonstration of sample data generators
    print("Sample Data Generators Demonstration")
    print("=" * 50)

    # Test basic GBM path generation
    print("\n1. Geometric Brownian Motion Paths:")
    print("-" * 40)

    generator = SampleDataGenerator(random_seed=42)

    # Generate GBM paths
    gbm_paths = generator.generate_gbm_paths(
        S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=5
    )

    print(f"Generated {gbm_paths.shape[0]} GBM paths with {gbm_paths.shape[1]} time steps")
    print(f"Final prices: {gbm_paths[:, -1].round(2)}")
    print(f"Price range: ${gbm_paths.min():.2f} - ${gbm_paths.max():.2f}")

    # Test jump diffusion
    print("\n2. Jump Diffusion Paths:")
    print("-" * 30)

    jump_paths = generator.generate_jump_diffusion_paths(
        S0=100, mu=0.05, sigma=0.2, lambda_jump=0.1, mu_jump=-0.1, sigma_jump=0.3,
        T=1.0, n_steps=252, n_paths=3
    )

    print(f"Generated {jump_paths.shape[0]} jump diffusion paths")
    print(f"Final prices: {jump_paths[:, -1].round(2)}")

    # Compare with GBM
    gbm_returns = np.diff(np.log(gbm_paths[0, :]))
    jump_returns = np.diff(np.log(jump_paths[0, :]))

    print(f"GBM volatility: {np.std(gbm_returns) * np.sqrt(252):.3f}")
    print(f"Jump diffusion volatility: {np.std(jump_returns) * np.sqrt(252):.3f}")

    # Test Heston paths
    print("\n3. Heston Stochastic Volatility Paths:")
    print("-" * 40)

    S_paths, v_paths = generator.generate_heston_paths(
        S0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, r=0.05,
        T=1.0, n_steps=252, n_paths=3
    )

    print(f"Generated Heston paths: {S_paths.shape[0]} price paths, {v_paths.shape[0]} volatility paths")
    print(f"Final prices: {S_paths[:, -1].round(2)}")
    print(f"Final volatilities: {np.sqrt(v_paths[:, -1]).round(3)}")
    print(f"Average volatility: {np.mean(np.sqrt(v_paths), axis=1).round(3)}")

    # Test correlated returns
    print("\n4. Correlated Multi-Asset Returns:")
    print("-" * 40)

    corr_generator = CorrelatedReturnsGenerator(random_seed=42)

    # Define assets and correlation
    n_assets = 4
    mean_returns = np.array([0.08, 0.06, 0.04, 0.10]) / 252  # Daily
    vol_returns = np.array([0.20, 0.15, 0.10, 0.25]) / np.sqrt(252)  # Daily

    # Create correlation matrix
    correlation = np.array([
        [1.0, 0.7, 0.3, 0.6],
        [0.7, 1.0, 0.4, 0.5],
        [0.3, 0.4, 1.0, 0.2],
        [0.6, 0.5, 0.2, 1.0]
    ])

    # Convert to covariance matrix
    cov_matrix = np.outer(vol_returns, vol_returns) * correlation

    # Generate correlated returns
    correlated_returns = corr_generator.generate_multivariate_normal_returns(
        mean_returns, cov_matrix, n_periods=252
    )

    print(f"Generated correlated returns: {correlated_returns.shape}")
    print(f"Realized correlation matrix:")
    realized_corr = np.corrcoef(correlated_returns.T)
    print(realized_corr.round(3))

    # Test GARCH returns
    print("\n5. GARCH(1,1) Returns:")
    print("-" * 25)

    garch_returns, garch_vol = corr_generator.generate_garch_returns(
        mu=0.001, omega=0.00001, alpha=0.1, beta=0.85, n_periods=500
    )

    print(f"Generated GARCH returns: {len(garch_returns)} observations")
    print(f"Return statistics:")
    print(f"  Mean: {np.mean(garch_returns):.4f}")
    print(f"  Std: {np.std(garch_returns):.4f}")
    print(f"  Skewness: {pd.Series(garch_returns).skew():.3f}")
    print(f"  Kurtosis: {pd.Series(garch_returns).kurtosis():.3f}")

    print(f"Volatility statistics:")
    print(f"  Mean vol: {np.mean(garch_vol):.4f}")
    print(f"  Vol of vol: {np.std(garch_vol):.4f}")

    # Test regime switching
    print("\n6. Regime Switching Returns:")
    print("-" * 30)

    regime_params = [
        {'mu': 0.002, 'sigma': 0.015},  # Bull regime
        {'mu': -0.001, 'sigma': 0.030}  # Bear regime
    ]

    transition_matrix = np.array([
        [0.95, 0.05],  # Bull to Bull/Bear
        [0.10, 0.90]   # Bear to Bull/Bear
    ])

    regime_returns, regime_states = corr_generator.generate_regime_switching_returns(
        regime_params, transition_matrix, n_periods=500
    )

    print(f"Generated regime switching returns: {len(regime_returns)} observations")
    print(f"Regime distribution: {np.bincount(regime_states) / len(regime_states)}")

    # Calculate regime-conditional statistics
    for regime in [0, 1]:
        regime_mask = regime_states == regime
        if np.sum(regime_mask) > 0:
            regime_rets = regime_returns[regime_mask]
            print(f"Regime {regime}: Mean={np.mean(regime_rets):.4f}, Std={np.std(regime_rets):.4f}")

    # Test volatility models
    print("\n7. Stochastic Volatility Models:")
    print("-" * 35)

    vol_generator = VolatilityModelGenerator(random_seed=42)

    # CIR volatility paths
    cir_vol_paths = vol_generator.generate_stochastic_volatility_paths(
        v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.4, T=1.0, n_steps=252, n_paths=3, model='cir'
    )

    print(f"CIR volatility paths: {cir_vol_paths.shape}")
    print(f"Mean volatility levels: {np.mean(np.sqrt(cir_vol_paths), axis=1).round(3)}")

    # Generate implied volatility surface
    print("\n8. Implied Volatility Surface:")
    print("-" * 35)

    S0 = 100
    r = 0.05
    T_range = np.array([0.25, 0.5, 1.0, 2.0])  # 3M, 6M, 1Y, 2Y
    K_range = np.linspace(80, 120, 9)  # Strikes from 80 to 120

    vol_params = {
        'base_vol': 0.20,
        'vol_vol': 0.08,
        'skew': -0.15,
        'term_structure': 0.03
    }

    iv_surface = vol_generator.generate_implied_volatility_surface(
        S0, r, T_range, K_range, vol_params
    )

    print(f"Generated IV surface: {iv_surface.shape} (strikes x maturities)")
    print(f"IV range: {iv_surface.min():.1%} - {iv_surface.max():.1%}")

    # Show sample of surface
    print("\nSample IV Surface (strikes vs maturities):")
    print("Strike\\Maturity", end="")
    for T in T_range:
        print(f"{T:8.2f}Y", end="")
    print()

    for i, K in enumerate(K_range[::2]):  # Show every other strike
        print(f"{K:8.0f}", end="")
        for j in range(len(T_range)):
            print(f"{iv_surface[i*2, j]:8.1%}", end="")
        print()

    print(f"\nSample data generators demonstration completed successfully!")
    print(f"Generated multiple types of realistic financial data for testing and research")