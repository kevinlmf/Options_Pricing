import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from scipy import stats
import warnings

class ForecastMetrics:
    """
    Comprehensive evaluation metrics for time series forecasting.

    Includes point forecast metrics, probabilistic metrics, and distribution-based evaluations.
    """

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Mean Absolute Error.

        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Sample weights

        Returns:
            MAE value
        """
        error = np.abs(y_true - y_pred)
        if sample_weight is not None:
            return np.average(error, weights=sample_weight)
        return np.mean(error)

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Mean Squared Error.

        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Sample weights

        Returns:
            MSE value
        """
        error = (y_true - y_pred) ** 2
        if sample_weight is not None:
            return np.average(error, weights=sample_weight)
        return np.mean(error)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Root Mean Squared Error.

        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Sample weights

        Returns:
            RMSE value
        """
        return np.sqrt(ForecastMetrics.mse(y_true, y_pred, sample_weight))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small constant to avoid division by zero

        Returns:
            MAPE value as percentage
        """
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            sMAPE value as percentage
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        denominator = np.where(denominator == 0, 1, denominator)
        return np.mean(np.abs(y_true - y_pred) / denominator) * 100

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
             seasonal_period: int = 1) -> float:
        """
        Mean Absolute Scaled Error.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data for naive forecast
            seasonal_period: Seasonal period for naive forecast

        Returns:
            MASE value
        """
        # Compute naive forecast MAE
        if seasonal_period == 1:
            naive_forecast_mae = np.mean(np.abs(np.diff(y_train)))
        else:
            naive_forecast_mae = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))

        # Avoid division by zero
        if naive_forecast_mae == 0:
            return np.nan

        return ForecastMetrics.mae(y_true, y_pred) / naive_forecast_mae

    @staticmethod
    def crps(y_true: np.ndarray, forecast_mean: np.ndarray,
             forecast_std: np.ndarray) -> float:
        """
        Continuous Ranked Probability Score for Gaussian forecasts.

        Args:
            y_true: True values
            forecast_mean: Forecast means
            forecast_std: Forecast standard deviations

        Returns:
            CRPS value
        """
        # Standardize
        z = (y_true - forecast_mean) / forecast_std

        # CRPS for standard normal distribution
        crps_std = forecast_std * (
            z * (2 * stats.norm.cdf(z) - 1) +
            2 * stats.norm.pdf(z) -
            1 / np.sqrt(np.pi)
        )

        return np.mean(crps_std)

    @staticmethod
    def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """
        Pinball loss (quantile loss).

        Args:
            y_true: True values
            y_pred: Predicted quantiles
            quantile: Quantile level (0-1)

        Returns:
            Pinball loss
        """
        error = y_true - y_pred
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

    @staticmethod
    def coverage_probability(y_true: np.ndarray, lower_bound: np.ndarray,
                           upper_bound: np.ndarray) -> float:
        """
        Coverage probability for prediction intervals.

        Args:
            y_true: True values
            lower_bound: Lower bounds of prediction intervals
            upper_bound: Upper bounds of prediction intervals

        Returns:
            Coverage probability (0-1)
        """
        coverage = (y_true >= lower_bound) & (y_true <= upper_bound)
        return np.mean(coverage)

    @staticmethod
    def interval_width(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
        """
        Average width of prediction intervals.

        Args:
            lower_bound: Lower bounds of prediction intervals
            upper_bound: Upper bounds of prediction intervals

        Returns:
            Average interval width
        """
        return np.mean(upper_bound - lower_bound)

    @staticmethod
    def winkler_score(y_true: np.ndarray, lower_bound: np.ndarray,
                     upper_bound: np.ndarray, alpha: float = 0.05) -> float:
        """
        Winkler score for prediction intervals.

        Args:
            y_true: True values
            lower_bound: Lower bounds of prediction intervals
            upper_bound: Upper bounds of prediction intervals
            alpha: Significance level (1-confidence_level)

        Returns:
            Winkler score
        """
        width = upper_bound - lower_bound
        penalty = np.zeros_like(width)

        # Penalty for under-coverage
        below = y_true < lower_bound
        penalty[below] = (2 / alpha) * (lower_bound[below] - y_true[below])

        # Penalty for over-coverage
        above = y_true > upper_bound
        penalty[above] = (2 / alpha) * (y_true[above] - upper_bound[above])

        return np.mean(width + penalty)

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional accuracy (percentage of correct direction predictions).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Directional accuracy as percentage
        """
        if len(y_true) < 2:
            return np.nan

        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        return np.mean(true_direction == pred_direction) * 100

    @staticmethod
    def theil_u_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Theil's U statistic.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Theil U statistic
        """
        if len(y_true) < 2:
            return np.nan

        # Naive forecast (random walk)
        naive_pred = y_true[:-1]
        actual_changes = y_true[1:]

        mse_model = np.mean((actual_changes - y_pred[1:]) ** 2)
        mse_naive = np.mean((actual_changes - naive_pred) ** 2)

        if mse_naive == 0:
            return np.nan

        return np.sqrt(mse_model / mse_naive)

    @classmethod
    def comprehensive_evaluation(cls, y_true: np.ndarray, y_pred: np.ndarray,
                               forecast_std: Optional[np.ndarray] = None,
                               lower_bound: Optional[np.ndarray] = None,
                               upper_bound: Optional[np.ndarray] = None,
                               y_train: Optional[np.ndarray] = None,
                               seasonal_period: int = 1,
                               alpha: float = 0.05) -> Dict[str, float]:
        """
        Comprehensive forecast evaluation with all available metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            forecast_std: Forecast standard deviations (for probabilistic metrics)
            lower_bound: Lower bounds of prediction intervals
            upper_bound: Upper bounds of prediction intervals
            y_train: Training data (for MASE)
            seasonal_period: Seasonal period
            alpha: Significance level

        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}

        # Point forecast metrics
        metrics['MAE'] = cls.mae(y_true, y_pred)
        metrics['MSE'] = cls.mse(y_true, y_pred)
        metrics['RMSE'] = cls.rmse(y_true, y_pred)
        metrics['MAPE'] = cls.mape(y_true, y_pred)
        metrics['sMAPE'] = cls.smape(y_true, y_pred)

        # Scaled metrics
        if y_train is not None:
            metrics['MASE'] = cls.mase(y_true, y_pred, y_train, seasonal_period)

        # Directional accuracy
        metrics['Directional_Accuracy'] = cls.directional_accuracy(y_true, y_pred)

        # Theil U statistic
        metrics['Theil_U'] = cls.theil_u_statistic(y_true, y_pred)

        # Probabilistic metrics
        if forecast_std is not None:
            metrics['CRPS'] = cls.crps(y_true, y_pred, forecast_std)

        # Interval-based metrics
        if lower_bound is not None and upper_bound is not None:
            metrics['Coverage_Probability'] = cls.coverage_probability(
                y_true, lower_bound, upper_bound
            )
            metrics['Average_Interval_Width'] = cls.interval_width(
                lower_bound, upper_bound
            )
            metrics['Winkler_Score'] = cls.winkler_score(
                y_true, lower_bound, upper_bound, alpha
            )

        return metrics

    @staticmethod
    def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Forecast bias (mean error).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Mean bias
        """
        return np.mean(y_pred - y_true)

    @staticmethod
    def forecast_variance_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Ratio of forecast variance to actual variance.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Variance ratio
        """
        var_true = np.var(y_true, ddof=1)
        var_pred = np.var(y_pred, ddof=1)

        if var_true == 0:
            return np.nan

        return var_pred / var_true