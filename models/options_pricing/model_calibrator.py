"""
Unified Model Calibration Module

Calibrates all pricing models (Black-Scholes, Heston, SABR, Local Vol, Multi-Agent)
to market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize, differential_evolution
import logging

from .heston import HestonModel
from .sabr import SABRModel
from .local_volatility import LocalVolatilityModel
from .multi_agent.models.three_layer_measures import ThreeLayerMeasureFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCalibrator:
    """
    Unified calibration framework for all option pricing models
    """

    def __init__(self, market_data: pd.DataFrame, spot_price: float, risk_free_rate: float):
        """
        Initialize Model Calibrator

        Parameters:
        -----------
        market_data : pd.DataFrame
            Market option data with columns:
            ['strike', 'years_to_expiry', 'option_type', 'mark_price', 'mark_iv']
        spot_price : float
            Current underlying spot price
        risk_free_rate : float
            Risk-free interest rate
        """
        self.market_data = market_data
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate

        self.calibrated_models = {}
        self.calibration_errors = {}

    def calibrate_heston(self,
                        initial_params: Optional[Dict] = None,
                        method: str = 'differential_evolution') -> Dict:
        """
        Calibrate Heston model to market data

        Parameters:
        -----------
        initial_params : dict, optional
            Initial parameter guesses
        method : str
            Optimization method: 'L-BFGS-B', 'differential_evolution'

        Returns:
        --------
        dict : Calibrated parameters and statistics
        """
        logger.info("Calibrating Heston model...")

        if initial_params is None:
            initial_params = {
                'v0': 0.04,
                'kappa': 2.0,
                'theta': 0.04,
                'sigma_v': 0.3,
                'rho': -0.5
            }

        # Prepare calibration data
        cal_data = self.market_data.copy()

        def objective(params):
            v0, kappa, theta, sigma_v, rho = params

            # Parameter constraints (soft)
            if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0:
                return 1e10
            if abs(rho) >= 1.0:
                return 1e10
            if 2 * kappa * theta < sigma_v**2:  # Feller condition
                return 1e10 + (sigma_v**2 - 2 * kappa * theta)**2

            total_error = 0.0
            count = 0

            for _, row in cal_data.iterrows():
                try:
                    from models.heston import HestonParameters

                    params = HestonParameters(
                        S0=self.spot_price,
                        K=row['strike'],
                        T=row['years_to_expiry'],
                        r=self.risk_free_rate,
                        q=0.0,
                        v0=v0,
                        kappa=kappa,
                        theta=theta,
                        xi=sigma_v,  # Note: HestonParameters uses 'xi' not 'sigma_v'
                        rho=rho
                    )

                    heston = HestonModel(params)
                    model_price = heston.option_price(option_type=row['option_type'])
                    market_price = row['mark_price']

                    # Weighted error (by vega to emphasize ATM options)
                    error = (model_price - market_price)**2
                    total_error += error
                    count += 1

                except Exception as e:
                    # Penalize invalid parameters
                    total_error += 1e6

            return total_error / max(count, 1)

        # Parameter bounds
        bounds = [
            (0.001, 1.0),    # v0
            (0.1, 10.0),     # kappa
            (0.001, 1.0),    # theta
            (0.01, 2.0),     # sigma_v
            (-0.99, 0.99)    # rho
        ]

        x0 = [initial_params['v0'], initial_params['kappa'],
              initial_params['theta'], initial_params['sigma_v'],
              initial_params['rho']]

        if method == 'differential_evolution':
            # Reduced iterations for faster calibration
            result = differential_evolution(objective, bounds, seed=42, maxiter=20, workers=1)
        else:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})

        calibrated_params = {
            'v0': result.x[0],
            'kappa': result.x[1],
            'theta': result.x[2],
            'sigma_v': result.x[3],
            'rho': result.x[4],
            'rmse': np.sqrt(result.fun),
            'success': result.success
        }

        self.calibrated_models['heston'] = calibrated_params
        self.calibration_errors['heston'] = result.fun

        logger.info(f"Heston calibration complete. RMSE: {calibrated_params['rmse']:.6f}")

        return calibrated_params

    def calibrate_sabr(self,
                      beta: float = 0.7,
                      initial_params: Optional[Dict] = None) -> Dict:
        """
        Calibrate SABR model to market data

        Parameters:
        -----------
        beta : float
            Beta parameter (usually fixed)
        initial_params : dict, optional
            Initial parameter guesses

        Returns:
        --------
        dict : Calibrated parameters
        """
        logger.info("Calibrating SABR model...")

        if initial_params is None:
            initial_params = {
                'alpha': 0.3,
                'rho': -0.3,
                'nu': 0.4
            }

        cal_data = self.market_data.copy()

        # Group by expiry for term-structure calibration
        expiries = cal_data['years_to_expiry'].unique()

        calibrated_by_expiry = {}

        for T in expiries:
            expiry_data = cal_data[cal_data['years_to_expiry'] == T]

            def objective(params):
                alpha, rho, nu = params

                # Parameter validation
                if alpha <= 0 or nu <= 0 or abs(rho) >= 1.0:
                    return 1e10

                total_error = 0.0

                for _, row in expiry_data.iterrows():
                    try:
                        F = self.spot_price * np.exp(self.risk_free_rate * T)

                        sabr = SABRModel(
                            F=F,
                            K=row['strike'],
                            T=T,
                            r=self.risk_free_rate,
                            alpha=alpha,
                            beta=beta,
                            rho=rho,
                            nu=nu
                        )

                        model_iv = sabr.implied_volatility_hagan()
                        market_iv = row['mark_iv']

                        error = (model_iv - market_iv)**2
                        total_error += error

                    except:
                        total_error += 1e6

                return total_error / len(expiry_data)

            bounds = [
                (0.01, 2.0),      # alpha
                (-0.99, 0.99),    # rho
                (0.01, 2.0)       # nu
            ]

            x0 = [initial_params['alpha'], initial_params['rho'], initial_params['nu']]

            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

            calibrated_by_expiry[T] = {
                'alpha': result.x[0],
                'beta': beta,
                'rho': result.x[1],
                'nu': result.x[2],
                'rmse': np.sqrt(result.fun)
            }

        # Average parameters across expiries
        avg_params = {
            'alpha': np.mean([p['alpha'] for p in calibrated_by_expiry.values()]),
            'beta': beta,
            'rho': np.mean([p['rho'] for p in calibrated_by_expiry.values()]),
            'nu': np.mean([p['nu'] for p in calibrated_by_expiry.values()]),
            'by_expiry': calibrated_by_expiry,
            'avg_rmse': np.mean([p['rmse'] for p in calibrated_by_expiry.values()])
        }

        self.calibrated_models['sabr'] = avg_params
        self.calibration_errors['sabr'] = avg_params['avg_rmse']**2

        logger.info(f"SABR calibration complete. Avg RMSE: {avg_params['avg_rmse']:.6f}")

        return avg_params

    def calibrate_local_volatility(self) -> Dict:
        """
        Calibrate Local Volatility model from implied volatility surface

        Returns:
        --------
        dict : Local volatility surface
        """
        logger.info("Calibrating Local Volatility model...")

        # Prepare volatility surface
        strikes = self.market_data['strike'].unique()
        maturities = self.market_data['years_to_expiry'].unique()

        # Create IV grid
        iv_grid = np.zeros((len(strikes), len(maturities)))

        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                mask = (self.market_data['strike'] == K) & \
                       (self.market_data['years_to_expiry'] == T)

                if mask.any():
                    iv_grid[i, j] = self.market_data[mask]['mark_iv'].mean()
                else:
                    # Interpolation needed
                    iv_grid[i, j] = 0.5  # Default

        # Initialize local vol model
        local_vol_model = LocalVolatilityModel(
            S=self.spot_price,
            r=self.risk_free_rate
        )

        # Compute local volatility from implied volatility
        local_vols = local_vol_model.compute_local_vol_from_implied(
            strikes, maturities, iv_grid
        )

        local_vol_model.set_local_vol_surface(strikes, maturities, local_vols)

        calibrated = {
            'strikes': strikes,
            'maturities': maturities,
            'local_vol_surface': local_vols,
            'model': local_vol_model
        }

        self.calibrated_models['local_volatility'] = calibrated

        logger.info("Local Volatility calibration complete.")

        return calibrated

    def calibrate_multi_agent(self, config: Optional[Dict] = None) -> Dict:
        """
        Calibrate Multi-Agent model to market data

        Parameters:
        -----------
        config : dict, optional
            Multi-agent configuration

        Returns:
        --------
        dict : Calibrated multi-agent framework
        """
        logger.info("Calibrating Multi-Agent model...")

        try:
            # Initialize three-layer measure framework
            framework = ThreeLayerMeasureFramework(config=config or {})

            # Prepare historical data
            historical_data = {
                'prices': self.market_data['mark_price'].values,
                'strikes': self.market_data['strike'].values,
                'maturities': self.market_data['years_to_expiry'].values,
                'implied_vols': self.market_data['mark_iv'].values
            }

            # Calibrate all measures
            framework.calibrate_all_measures(
                historical_data=historical_data,
                current_market_equilibrium={'spot_price': self.spot_price},
                risk_free_rate=self.risk_free_rate
            )

            calibrated = {
                'framework': framework,
                'config': config,
                'calibration_metrics': {
                    'num_agents': framework.config.get('total_agents', 20),
                    'market_regimes': ['stable', 'unstable', 'stressed']
                }
            }

            self.calibrated_models['multi_agent'] = calibrated

            logger.info("Multi-Agent calibration complete.")

            return calibrated

        except Exception as e:
            logger.error(f"Multi-Agent calibration failed: {e}")
            return {}

    def calibrate_all_models(self) -> Dict[str, Any]:
        """
        Calibrate all models to market data

        Returns:
        --------
        dict : All calibrated models
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive model calibration")
        logger.info("=" * 60)

        results = {}

        # Calibrate Heston
        try:
            results['heston'] = self.calibrate_heston()
        except Exception as e:
            logger.error(f"Heston calibration failed: {e}")
            results['heston'] = None

        # Calibrate SABR
        try:
            results['sabr'] = self.calibrate_sabr()
        except Exception as e:
            logger.error(f"SABR calibration failed: {e}")
            results['sabr'] = None

        # Calibrate Local Vol
        try:
            results['local_volatility'] = self.calibrate_local_volatility()
        except Exception as e:
            logger.error(f"Local Volatility calibration failed: {e}")
            results['local_volatility'] = None

        # Calibrate Multi-Agent
        try:
            results['multi_agent'] = self.calibrate_multi_agent()
        except Exception as e:
            logger.error(f"Multi-Agent calibration failed: {e}")
            results['multi_agent'] = None

        logger.info("=" * 60)
        logger.info("Calibration complete for all models")
        logger.info("=" * 60)

        return results

    def compare_calibration_quality(self) -> pd.DataFrame:
        """
        Compare calibration quality across models

        Returns:
        --------
        pd.DataFrame : Comparison metrics
        """
        comparison = []

        for model_name, error in self.calibration_errors.items():
            rmse = np.sqrt(error)

            comparison.append({
                'Model': model_name.upper(),
                'RMSE': rmse,
                'R²': 1 - error / np.var(self.market_data['mark_price']),
                'Calibration_Success': model_name in self.calibrated_models
            })

        return pd.DataFrame(comparison)

    def get_calibrated_model(self, model_name: str) -> Optional[Any]:
        """
        Get calibrated model by name

        Parameters:
        -----------
        model_name : str
            Model name: 'heston', 'sabr', 'local_volatility', 'multi_agent'

        Returns:
        --------
        Calibrated model object or parameters
        """
        return self.calibrated_models.get(model_name)
