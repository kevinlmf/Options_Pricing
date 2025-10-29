"""
Model Validation Framework

Implementation of comprehensive model validation and backtesting tools.
Follows the mathematical framework: Pure Math → Applied Math → Model Validation

Mathematical Foundation:
- Pure Math: Hypothesis testing, statistical inference, probability theory
- Applied Math: Time series validation, cross-validation, bootstrap methods
- Model Validation: Out-of-sample testing, performance evaluation, model comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from datetime import datetime, timedelta


class ModelValidator(ABC):
    """Abstract base class for model validators"""

    @abstractmethod
    def validate_model(self, model, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validate a model and return validation results"""
        pass

    @abstractmethod
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report"""
        pass


class BacktestFramework:
    """
    Comprehensive backtesting framework for derivatives and risk models.
    """

    def __init__(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initialize backtesting framework.

        Args:
            start_date: Backtesting start date
            end_date: Backtesting end date
        """
        self.start_date = start_date
        self.end_date = end_date
        self.backtest_results = {}

    def time_series_split_backtest(self, model, data: pd.DataFrame,
                                  target_col: str, feature_cols: List[str],
                                  n_splits: int = 5, test_size: int = 60) -> Dict[str, Any]:
        """
        Perform time series cross-validation backtesting.

        Args:
            model: Model to backtest (must have fit/predict methods)
            data: Input data
            target_col: Target column name
            feature_cols: Feature column names
            n_splits: Number of CV splits
            test_size: Size of test set for each split

        Returns:
            Backtesting results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        results = {
            'split_results': [],
            'predictions': [],
            'actuals': [],
            'metrics': {}
        }

        X = data[feature_cols].values
        y = data[target_col].values

        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Store results
            split_result = {
                'split': i,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'predictions': y_pred,
                'actuals': y_test,
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }

            results['split_results'].append(split_result)
            results['predictions'].extend(y_pred)
            results['actuals'].extend(y_test)

        # Calculate overall metrics
        all_predictions = np.array(results['predictions'])
        all_actuals = np.array(results['actuals'])

        results['metrics'] = {
            'overall_mse': mean_squared_error(all_actuals, all_predictions),
            'overall_mae': mean_absolute_error(all_actuals, all_predictions),
            'overall_rmse': np.sqrt(mean_squared_error(all_actuals, all_predictions)),
            'mean_split_mse': np.mean([s['mse'] for s in results['split_results']]),
            'std_split_mse': np.std([s['mse'] for s in results['split_results']]),
            'directional_accuracy': self._calculate_directional_accuracy(all_actuals, all_predictions)
        }

        return results

    def rolling_window_backtest(self, model_func: Callable, data: pd.DataFrame,
                               window_size: int = 252, step_size: int = 1,
                               prediction_horizon: int = 1) -> Dict[str, Any]:
        """
        Perform rolling window backtesting.

        Args:
            model_func: Function that takes data and returns trained model
            data: Input data
            window_size: Size of training window
            step_size: Step size for rolling window
            prediction_horizon: Number of periods ahead to predict

        Returns:
            Rolling backtest results
        """
        n_obs = len(data)
        results = {
            'predictions': [],
            'actuals': [],
            'timestamps': [],
            'training_periods': []
        }

        for i in range(window_size, n_obs - prediction_horizon + 1, step_size):
            # Training data
            train_data = data.iloc[i - window_size:i]

            # Test data
            test_data = data.iloc[i:i + prediction_horizon]

            try:
                # Train model
                model = model_func(train_data)

                # Make prediction
                prediction = model.predict(test_data)
                actual = test_data.iloc[-1]['target'] if 'target' in test_data.columns else test_data.iloc[-1, 0]

                results['predictions'].append(prediction)
                results['actuals'].append(actual)
                results['timestamps'].append(data.index[i + prediction_horizon - 1])
                results['training_periods'].append((data.index[i - window_size], data.index[i - 1]))

            except Exception as e:
                warnings.warn(f"Model training failed at step {i}: {e}")
                continue

        # Calculate metrics
        if results['predictions']:
            predictions = np.array(results['predictions'])
            actuals = np.array(results['actuals'])

            results['metrics'] = {
                'mse': mean_squared_error(actuals, predictions),
                'mae': mean_absolute_error(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                'directional_accuracy': self._calculate_directional_accuracy(actuals, predictions),
                'correlation': np.corrcoef(actuals, predictions)[0, 1],
                'hit_rate': np.mean(np.abs(actuals - predictions) < np.std(actuals))
            }

        return results

    def option_pricing_backtest(self, pricing_model, market_data: pd.DataFrame,
                               option_data: pd.DataFrame, validation_window: int = 30) -> Dict[str, Any]:
        """
        Backtest option pricing models against market prices.

        Args:
            pricing_model: Option pricing model
            market_data: Market data (underlying prices, rates, etc.)
            option_data: Option market data with strikes, expiries, market prices
            validation_window: Window for model recalibration

        Returns:
            Option pricing backtest results
        """
        results = {
            'pricing_errors': [],
            'implied_vol_errors': [],
            'delta_errors': [],
            'timestamps': [],
            'strikes': [],
            'expiries': [],
            'option_types': []
        }

        # Group by date
        for date in option_data.index.unique():
            date_options = option_data.loc[date]

            if isinstance(date_options, pd.Series):
                date_options = date_options.to_frame().T

            # Get corresponding market data
            market_slice = market_data.loc[market_data.index <= date].tail(validation_window)

            if len(market_slice) == 0:
                continue

            # Current market conditions
            S0 = market_slice['Close'].iloc[-1]
            r = market_slice.get('RiskFreeRate', 0.05).iloc[-1]
            q = market_slice.get('DividendYield', 0.0).iloc[-1]

            for _, option in date_options.iterrows():
                try:
                    K = option['Strike']
                    T = option['TimeToExpiry']
                    market_price = option['MarketPrice']
                    option_type = option['Type']
                    market_iv = option.get('ImpliedVol', None)

                    # Update model parameters
                    pricing_model.params.S0 = S0
                    pricing_model.params.K = K
                    pricing_model.params.T = T
                    pricing_model.params.r = r
                    pricing_model.params.q = q

                    # Calculate model price
                    model_price = pricing_model.option_price(option_type)

                    # Calculate errors
                    price_error = model_price - market_price
                    relative_error = price_error / market_price

                    results['pricing_errors'].append({
                        'absolute_error': price_error,
                        'relative_error': relative_error,
                        'market_price': market_price,
                        'model_price': model_price
                    })

                    # Implied volatility comparison if available
                    if market_iv is not None:
                        model_iv = pricing_model.implied_volatility(market_price, option_type)
                        iv_error = model_iv - market_iv
                        results['implied_vol_errors'].append(iv_error)

                    # Greeks comparison if available
                    if 'MarketDelta' in option:
                        model_delta = pricing_model.delta(option_type)
                        delta_error = model_delta - option['MarketDelta']
                        results['delta_errors'].append(delta_error)

                    results['timestamps'].append(date)
                    results['strikes'].append(K)
                    results['expiries'].append(T)
                    results['option_types'].append(option_type)

                except Exception as e:
                    warnings.warn(f"Option pricing failed for {option_type} option: {e}")
                    continue

        # Calculate summary metrics
        if results['pricing_errors']:
            pricing_errors = [e['absolute_error'] for e in results['pricing_errors']]
            relative_errors = [e['relative_error'] for e in results['pricing_errors']]

            results['summary_metrics'] = {
                'mean_absolute_error': np.mean(np.abs(pricing_errors)),
                'rmse': np.sqrt(np.mean(np.array(pricing_errors)**2)),
                'mean_relative_error': np.mean(relative_errors),
                'rmse_relative': np.sqrt(np.mean(np.array(relative_errors)**2))
            }

            if results['implied_vol_errors']:
                results['summary_metrics']['iv_mae'] = np.mean(np.abs(results['implied_vol_errors']))
                results['summary_metrics']['iv_rmse'] = np.sqrt(np.mean(np.array(results['implied_vol_errors'])**2))

        return results

    def risk_model_backtest(self, risk_model, returns_data: pd.DataFrame,
                           confidence_levels: List[float] = [0.01, 0.05, 0.10],
                           window_size: int = 250) -> Dict[str, Any]:
        """
        Backtest risk models (VaR, CVaR) against actual losses.

        Args:
            risk_model: Risk model with calculate_var/calculate_cvar methods
            returns_data: Historical returns data
            confidence_levels: List of confidence levels to test
            window_size: Window size for risk estimation

        Returns:
            Risk model backtest results
        """
        results = {
            'confidence_levels': confidence_levels,
            'backtest_results': {},
            'summary_metrics': {}
        }

        for confidence_level in confidence_levels:
            level_results = {
                'var_estimates': [],
                'cvar_estimates': [],
                'actual_losses': [],
                'var_violations': [],
                'cvar_violations': [],
                'timestamps': []
            }

            # Rolling window backtesting
            for i in range(window_size, len(returns_data)):
                # Training window
                train_data = returns_data.iloc[i - window_size:i]

                # Actual loss for next period
                actual_return = returns_data.iloc[i]
                actual_loss = -actual_return if hasattr(actual_return, '__iter__') else -actual_return

                # Estimate risk measures
                try:
                    risk_model.confidence_level = confidence_level
                    var_estimate = risk_model.calculate_var(train_data.values if hasattr(train_data, 'values') else train_data)

                    if hasattr(risk_model, 'calculate_cvar'):
                        cvar_estimate = risk_model.calculate_cvar(train_data.values if hasattr(train_data, 'values') else train_data)
                    else:
                        cvar_estimate = None

                    # Check violations
                    var_violation = actual_loss > var_estimate
                    cvar_violation = actual_loss > cvar_estimate if cvar_estimate is not None else False

                    # Store results
                    level_results['var_estimates'].append(var_estimate)
                    if cvar_estimate is not None:
                        level_results['cvar_estimates'].append(cvar_estimate)
                    level_results['actual_losses'].append(actual_loss)
                    level_results['var_violations'].append(var_violation)
                    level_results['cvar_violations'].append(cvar_violation)
                    level_results['timestamps'].append(returns_data.index[i])

                except Exception as e:
                    warnings.warn(f"Risk estimation failed at step {i}: {e}")
                    continue

            # Calculate backtest statistics
            if level_results['var_estimates']:
                var_violations = np.array(level_results['var_violations'])
                violation_rate = np.mean(var_violations)
                expected_rate = confidence_level

                # Kupiec test
                n_violations = np.sum(var_violations)
                n_obs = len(var_violations)

                if n_violations > 0 and n_violations < n_obs:
                    lr_stat = -2 * (n_violations * np.log(confidence_level) +
                                   (n_obs - n_violations) * np.log(1 - confidence_level) -
                                   n_violations * np.log(violation_rate) -
                                   (n_obs - n_violations) * np.log(1 - violation_rate))
                    kupiec_pvalue = 1 - stats.chi2.cdf(lr_stat, 1)
                else:
                    kupiec_pvalue = 0

                level_results['backtest_stats'] = {
                    'violation_rate': violation_rate,
                    'expected_rate': expected_rate,
                    'kupiec_pvalue': kupiec_pvalue,
                    'kupiec_test_passed': kupiec_pvalue > 0.05
                }

            results['backtest_results'][confidence_level] = level_results

        return results

    def _calculate_directional_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate directional accuracy for predictions"""
        if len(actual) < 2 or len(predicted) < 2:
            return np.nan

        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))

        correct_directions = actual_direction == predicted_direction
        return np.mean(correct_directions)


class ValidationMetrics:
    """
    Collection of validation metrics for different types of models.
    """

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression validation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification validation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score, log_loss
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except:
                pass

        return metrics

    @staticmethod
    def time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate time series specific validation metrics"""
        basic_metrics = ValidationMetrics.regression_metrics(y_true, y_pred)

        # Directional accuracy
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            predicted_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == predicted_direction)
        else:
            directional_accuracy = np.nan

        # Theil's U statistic
        mse = basic_metrics['mse']
        naive_mse = np.mean((y_true[1:] - y_true[:-1]) ** 2) if len(y_true) > 1 else np.inf
        theil_u = np.sqrt(mse) / np.sqrt(naive_mse) if naive_mse > 0 else np.inf

        basic_metrics.update({
            'directional_accuracy': directional_accuracy,
            'theil_u': theil_u
        })

        return basic_metrics

    @staticmethod
    def risk_model_metrics(actual_losses: np.ndarray, var_estimates: np.ndarray,
                          confidence_level: float) -> Dict[str, float]:
        """Calculate risk model validation metrics"""
        violations = actual_losses > var_estimates
        violation_rate = np.mean(violations)
        expected_rate = confidence_level

        # Kupiec test
        n_violations = np.sum(violations)
        n_obs = len(violations)

        if 0 < n_violations < n_obs:
            lr_stat = -2 * (n_violations * np.log(confidence_level) +
                           (n_obs - n_violations) * np.log(1 - confidence_level) -
                           n_violations * np.log(violation_rate) -
                           (n_obs - n_violations) * np.log(1 - violation_rate))
            kupiec_pvalue = 1 - stats.chi2.cdf(lr_stat, 1)
        else:
            kupiec_pvalue = 0

        # Average exceedance size
        if n_violations > 0:
            exceedances = actual_losses[violations] - var_estimates[violations]
            avg_exceedance = np.mean(exceedances)
            max_exceedance = np.max(exceedances)
        else:
            avg_exceedance = 0
            max_exceedance = 0

        return {
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_statistic': lr_stat if 0 < n_violations < n_obs else 0,
            'kupiec_pvalue': kupiec_pvalue,
            'test_passed': kupiec_pvalue > 0.05,
            'avg_exceedance': avg_exceedance,
            'max_exceedance': max_exceedance
        }


if __name__ == "__main__":
    # Demonstration of model validation framework
    print("Model Validation Framework Demonstration")
    print("=" * 60)

    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_obs = len(dates)

    # Sample returns data
    returns = np.random.normal(0.0005, 0.02, n_obs)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'returns': returns,
        'prices': prices,
        'volatility': pd.Series(returns).rolling(30).std(),
        'target': np.roll(returns, -1)  # Next day return as target
    }, index=dates)

    data = data.dropna()

    print(f"Generated sample data: {len(data)} observations")

    # Test backtesting framework
    backtest_framework = BacktestFramework()

    print("\n1. Rolling Window Backtest:")
    print("-" * 30)

    # Simple linear model for demonstration
    from sklearn.linear_model import LinearRegression

    def simple_model_func(train_data):
        model = LinearRegression()
        # Use lagged returns as features
        X = train_data[['returns']].shift(1).dropna()
        y = train_data['target'][1:]
        model.fit(X, y)
        return model

    # Note: This is a simplified example - in practice you'd have more sophisticated models
    print("Rolling window backtest would be performed here with proper model implementation")

    # Test validation metrics
    print("\n2. Validation Metrics:")
    print("-" * 25)

    # Generate sample predictions for demonstration
    y_true = data['target'][:100].values
    y_pred = y_true + np.random.normal(0, 0.005, len(y_true))  # Add some prediction error

    # Regression metrics
    regression_metrics = ValidationMetrics.regression_metrics(y_true, y_pred)
    print("Regression Metrics:")
    for metric, value in regression_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Time series metrics
    ts_metrics = ValidationMetrics.time_series_metrics(y_true, y_pred)
    print("\nTime Series Metrics:")
    print(f"  Directional Accuracy: {ts_metrics['directional_accuracy']:.3f}")
    print(f"  Theil U: {ts_metrics['theil_u']:.3f}")

    # Risk model metrics
    print("\n3. Risk Model Validation:")
    print("-" * 30)

    # Generate sample VaR estimates
    actual_losses = -data['returns'][:252].values  # Convert returns to losses
    var_estimates = np.abs(np.random.normal(0.02, 0.005, len(actual_losses)))  # Sample VaR estimates

    risk_metrics = ValidationMetrics.risk_model_metrics(actual_losses, var_estimates, confidence_level=0.05)
    print("VaR Model Metrics (5% confidence level):")
    for metric, value in risk_metrics.items():
        if isinstance(value, bool):
            print(f"  {metric}: {value}")
        elif isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")

    print("\n4. Backtesting Statistics:")
    print("-" * 30)

    # Demonstrate backtesting statistics
    violations = actual_losses > var_estimates
    print(f"Total observations: {len(actual_losses)}")
    print(f"VaR violations: {np.sum(violations)}")
    print(f"Violation rate: {np.mean(violations):.2%}")
    print(f"Expected rate: 5.00%")
    print(f"Kupiec test passed: {risk_metrics['test_passed']}")

    if np.sum(violations) > 0:
        violation_dates = np.where(violations)[0]
        print(f"Largest violation: {np.max(actual_losses[violations] - var_estimates[violations]):.4f}")

    print("\n5. Model Comparison Example:")
    print("-" * 35)

    # Compare two models
    model1_pred = y_true + np.random.normal(0, 0.003, len(y_true))
    model2_pred = y_true + np.random.normal(0, 0.007, len(y_true))

    model1_metrics = ValidationMetrics.regression_metrics(y_true, model1_pred)
    model2_metrics = ValidationMetrics.regression_metrics(y_true, model2_pred)

    print("Model Comparison (lower is better for error metrics):")
    print(f"{'Metric':<15} {'Model 1':<12} {'Model 2':<12} {'Winner'}")
    print("-" * 50)

    for metric in ['rmse', 'mae', 'mape']:
        val1 = model1_metrics[metric]
        val2 = model2_metrics[metric]
        winner = "Model 1" if val1 < val2 else "Model 2"
        print(f"{metric.upper():<15} {val1:<12.6f} {val2:<12.6f} {winner}")

    print(f"{'R²':<15} {model1_metrics['r2']:<12.6f} {model2_metrics['r2']:<12.6f} {'Model 1' if model1_metrics['r2'] > model2_metrics['r2'] else 'Model 2'}")

    print(f"\nModel validation framework demonstration completed successfully!")
    print(f"Framework provides comprehensive tools for validating derivatives and risk models")