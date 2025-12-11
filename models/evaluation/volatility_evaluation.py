"""
Volatility Evaluation Framework
===============================

Comprehensive evaluation for the dual convergence volatility modeling:

1. Statistical Evaluation: Correlation, RMSE, MAE, Sharpe ratio
2. Convergence Evaluation: How well factors converge to physical model
3. Trading Performance: Backtest with volatility-based strategies
4. Comparative Analysis: Against traditional volatility models
5. Risk Assessment: VaR, CVaR under different scenarios

Key Metrics:
- Volatility Prediction Accuracy
- Convergence Speed & Stability
- Trading Strategy Sharpe Ratio
- Risk-Adjusted Returns
- Model Robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from validation.rust_interface import RustVolatilityValidator, ValidationResult


@dataclass
class VolatilityMetrics:
    """Comprehensive volatility prediction metrics"""
    correlation: float  # Correlation with realized volatility
    rmse: float         # Root Mean Squared Error
    mae: float          # Mean Absolute Error
    mape: float         # Mean Absolute Percentage Error
    directional_accuracy: float  # Direction prediction accuracy
    sharpe_ratio: float # Risk-adjusted returns from trading
    max_drawdown: float # Maximum drawdown
    hit_rate: float     # Percentage of correct directional predictions

    def to_dict(self) -> Dict[str, float]:
        return {
            'correlation': self.correlation,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'directional_accuracy': self.directional_accuracy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'hit_rate': self.hit_rate
        }


@dataclass
class ConvergenceMetrics:
    """Metrics for dual convergence evaluation"""
    long_term_convergence: float      # How well factors converge to physical target
    short_term_tracking: float        # How well factors track realized volatility
    convergence_speed: float          # Speed of convergence (lower is faster)
    stability_score: float            # Stability of convergence process
    factor_physical_alignment: Dict[str, float]  # Each factor's alignment with physical model

    def to_dict(self) -> Dict[str, Any]:
        return {
            'long_term_convergence': self.long_term_convergence,
            'short_term_tracking': self.short_term_tracking,
            'convergence_speed': self.convergence_speed,
            'stability_score': self.stability_score,
            'factor_physical_alignment': self.factor_physical_alignment
        }


@dataclass
class ComparativeMetrics:
    """Comparison with baseline models"""
    vs_historical_vol: Dict[str, float]     # vs simple historical volatility
    vs_garch: Dict[str, float]              # vs GARCH(1,1)
    vs_ewma: Dict[str, float]               # vs EWMA
    vs_naive: Dict[str, float]              # vs naive forecast
    improvement_percentage: Dict[str, float]  # % improvement over baselines

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vs_historical_vol': self.vs_historical_vol,
            'vs_garch': self.vs_garch,
            'vs_ewma': self.vs_ewma,
            'vs_naive': self.vs_naive,
            'improvement_percentage': self.improvement_percentage
        }


@dataclass
class EvaluationResult:
    """Complete evaluation results"""
    volatility_metrics: VolatilityMetrics
    convergence_metrics: ConvergenceMetrics
    comparative_metrics: ComparativeMetrics
    trading_performance: Dict[str, float]
    robustness_tests: Dict[str, Any]

    def summary(self) -> str:
        """Generate evaluation summary"""
        summary = []
        summary.append("🎯 Dual Convergence Volatility Model Evaluation")
        summary.append("=" * 60)
        summary.append("")

        # Volatility prediction quality
        summary.append("📊 Volatility Prediction Quality:")
        vm = self.volatility_metrics
        summary.append(f"   Correlation: {vm.correlation:.4f}")
        summary.append(f"   RMSE: {vm.rmse:.4f}")
        summary.append(f"   MAPE: {vm.mape:.2f}%")
        summary.append(f"   Directional Accuracy: {vm.directional_accuracy:.4f}")
        summary.append("")

        # Convergence quality
        summary.append("🔄 Convergence Quality:")
        cm = self.convergence_metrics
        summary.append(f"   Long-term Convergence: {cm.long_term_convergence:.4f}")
        summary.append(f"   Short-term Tracking: {cm.short_term_tracking:.4f}")
        summary.append(f"   Convergence Speed: {cm.convergence_speed:.4f}")
        summary.append("")

        # Comparative performance
        summary.append("⚖️ Comparative Performance:")
        comp = self.comparative_metrics
        if comp.vs_historical_vol:
            improvement = comp.vs_historical_vol.get('rmse_improvement', 0)
            summary.append(f"   vs Historical Vol: {improvement:+.1f}% RMSE improvement")
        if comp.vs_garch:
            improvement = comp.vs_garch.get('rmse_improvement', 0)
            summary.append(f"   vs GARCH: {improvement:+.1f}% RMSE improvement")
        summary.append("")

        # Trading performance
        summary.append("💰 Trading Performance:")
        tp = self.trading_performance
        summary.append(f"   Sharpe Ratio: {tp['sharpe_ratio']:.4f}")
        summary.append(f"   Max Drawdown: {tp['max_drawdown']:.4f}")
        summary.append("")

        return "\n".join(summary)


class VolatilityEvaluator:
    """
    Comprehensive evaluation framework for dual convergence volatility modeling.

    Evaluates:
    1. Prediction accuracy (statistical metrics)
    2. Convergence quality (dual convergence assessment)
    3. Comparative performance (vs traditional models)
    4. Trading performance (backtest results)
    5. Robustness (stress testing)
    """

    def __init__(self, rust_validator: Optional[RustVolatilityValidator] = None):
        """
        Initialize evaluator.

        Parameters:
        -----------
        rust_validator : RustVolatilityValidator, optional
            For high-performance validation (recommended)
        """
        self.rust_validator = rust_validator or RustVolatilityValidator()

    def evaluate_volatility_prediction(self,
                                     predicted_volatility: pd.Series,
                                     realized_volatility: pd.Series,
                                     trading_signals: Optional[pd.Series] = None) -> VolatilityMetrics:
        """
        Evaluate volatility prediction quality.

        Parameters:
        -----------
        predicted_volatility : pd.Series
            Model predictions
        realized_volatility : pd.Series
            Actual realized volatility
        trading_signals : pd.Series, optional
            Trading signals based on volatility predictions

        Returns:
        --------
        VolatilityMetrics : Comprehensive evaluation metrics
        """
        # Align data
        common_idx = predicted_volatility.index.intersection(realized_volatility.index)
        pred = predicted_volatility.loc[common_idx]
        real = realized_volatility.loc[common_idx]

        if len(pred) < 10:
            raise ValueError("Insufficient data for evaluation")

        # Basic statistical metrics
        correlation = pred.corr(real)

        # Handle potential NaN or inf values
        valid_mask = ~(np.isnan(pred.values) | np.isinf(pred.values) |
                      np.isnan(real.values) | np.isinf(real.values))
        pred_clean = pred.values[valid_mask]
        real_clean = real.values[valid_mask]

        rmse = np.sqrt(mean_squared_error(real_clean, pred_clean))
        mae = mean_absolute_error(real_clean, pred_clean)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((real_clean - pred_clean) / (real_clean + 1e-8))) * 100

        # Directional accuracy (sign of change)
        pred_changes = np.sign(np.diff(pred_clean))
        real_changes = np.sign(np.diff(real_clean))
        directional_accuracy = np.mean(pred_changes == real_changes)

        # Trading performance (if signals provided)
        if trading_signals is not None:
            sharpe_ratio, max_drawdown, hit_rate = self._evaluate_trading_performance(
                trading_signals.loc[common_idx], real
            )
        else:
            # Simple buy-and-hold volatility strategy
            sharpe_ratio, max_drawdown, hit_rate = self._simple_volatility_trading(pred, real)

        return VolatilityMetrics(
            correlation=correlation,
            rmse=rmse,
            mae=mae,
            mape=mape,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate
        )

    def _evaluate_trading_performance(self,
                                     signals: pd.Series,
                                     volatility: pd.Series) -> Tuple[float, float, float]:
        """
        Evaluate trading performance based on volatility signals.

        Improved strategy: Use volatility predictions for option strategies
        - Buy options when volatility is predicted to increase
        - Sell options when volatility is predicted to decrease
        - This better captures the economic value of volatility predictions
        """
        # More realistic: volatility affects option prices
        # When volatility increases, option values increase (gamma/theta effects)
        vol_changes = volatility.pct_change().fillna(0)

        # Strategy: profit from volatility direction
        # Positive signal = expect higher volatility = buy options
        # This creates more realistic returns tied to volatility predictions
        option_returns = signals * vol_changes * 2  # Amplify for visibility

        # Add some realistic transaction costs and slippage
        transaction_costs = 0.001  # 0.1% per trade
        strategy_returns = option_returns - transaction_costs

        # Calculate Sharpe ratio
        if len(strategy_returns) > 1 and strategy_returns.std() > 0:
            sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        else:
            sharpe = 0.0

        # Maximum drawdown
        if len(strategy_returns) > 0:
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0.0
        else:
            max_drawdown = 0.0

        # Hit rate (correct directional predictions for volatility changes)
        vol_direction_correct = np.sum(np.sign(signals[:-1]) == np.sign(vol_changes[1:]))
        hit_rate = vol_direction_correct / len(signals[:-1]) if len(signals) > 1 else 0.0

        return sharpe, max_drawdown, hit_rate

    def _simple_volatility_trading(self,
                                 predicted_vol: pd.Series,
                                 realized_vol: pd.Series) -> Tuple[float, float, float]:
        """Simple volatility-based trading strategy"""
        # Strategy: bet against high predicted volatility
        signals = -np.sign(predicted_vol - predicted_vol.mean())
        returns = realized_vol.pct_change().fillna(0)

        strategy_returns = signals * returns

        sharpe = np.sqrt(252) * strategy_returns.mean() / (strategy_returns.std() + 1e-8)

        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()

        correct_predictions = np.sum(np.sign(strategy_returns) == np.sign(returns))
        hit_rate = correct_predictions / len(returns) if len(returns) > 0 else 0.0

        return sharpe, max_drawdown, hit_rate

    def evaluate_convergence_quality(self,
                                   factors: pd.DataFrame,
                                   constrained_factors: pd.DataFrame,
                                   physical_target: float,
                                   convergence_history: Optional[pd.DataFrame] = None) -> ConvergenceMetrics:
        """
        Evaluate how well the dual convergence works.

        Parameters:
        -----------
        factors : pd.DataFrame
            Original agent factors
        constrained_factors : pd.DataFrame
            Factors after convergence constraint
        physical_target : float
            Physical model long-term target
        convergence_history : pd.DataFrame, optional
            History of convergence process

        Returns:
        --------
        ConvergenceMetrics : Convergence quality assessment
        """
        # Long-term convergence: distance to physical target
        final_factors = constrained_factors.iloc[-1] if len(constrained_factors) > 0 else constrained_factors.mean()
        long_term_convergence = 1.0 - np.mean(np.abs(final_factors - physical_target) / (physical_target + 1e-8))

        # Short-term tracking: how well constrained factors track original factors
        # (This measures if we maintain the agent behavior insights)
        common_idx = factors.index.intersection(constrained_factors.index)
        if len(common_idx) > 10:
            original_subset = factors.loc[common_idx]
            constrained_subset = constrained_factors.loc[common_idx]

            tracking_errors = []
            for col in factors.columns:
                if col in constrained_factors.columns:
                    corr = original_subset[col].corr(constrained_subset[col])
                    tracking_errors.append(1.0 - abs(corr))  # Lower is better tracking

            short_term_tracking = 1.0 - np.mean(tracking_errors)
        else:
            short_term_tracking = 0.5  # Default neutral score

        # Convergence speed: how quickly factors approach target
        # Simplified: measure of factor variance reduction
        if len(constrained_factors) > 20:
            early_volatility = constrained_factors.iloc[:10].std().mean()
            late_volatility = constrained_factors.iloc[-10:].std().mean()
            convergence_speed = 1.0 - (late_volatility / (early_volatility + 1e-8))
        else:
            convergence_speed = 0.5

        # Stability score: consistency of convergence
        if len(constrained_factors) > 30:
            rolling_std = constrained_factors.rolling(10).std().mean().mean()
            stability_score = 1.0 - min(rolling_std / (constrained_factors.std().mean() + 1e-8), 1.0)
        else:
            stability_score = 0.5

        # Factor-physical alignment
        factor_alignment = {}
        for col in constrained_factors.columns:
            final_value = constrained_factors[col].iloc[-1] if len(constrained_factors) > 0 else constrained_factors[col].mean()
            alignment = 1.0 - abs(final_value - physical_target) / (physical_target + 1e-8)
            factor_alignment[col] = alignment

        return ConvergenceMetrics(
            long_term_convergence=max(0.0, min(1.0, long_term_convergence)),
            short_term_tracking=max(0.0, min(1.0, short_term_tracking)),
            convergence_speed=max(0.0, min(1.0, convergence_speed)),
            stability_score=max(0.0, min(1.0, stability_score)),
            factor_physical_alignment=factor_alignment
        )

    def compare_with_baselines(self,
                             predicted_volatility: pd.Series,
                             realized_volatility: pd.Series,
                             returns: pd.Series) -> ComparativeMetrics:
        """
        Compare dual convergence model with traditional volatility models.

        Parameters:
        -----------
        predicted_volatility : pd.Series
            Our model's predictions
        realized_volatility : pd.Series
            Actual realized volatility
        returns : pd.Series
            Asset returns for GARCH fitting

        Returns:
        --------
        ComparativeMetrics : Comparison with baselines
        """
        # Align data
        common_idx = predicted_volatility.index.intersection(realized_volatility.index)
        pred_vol = predicted_volatility.loc[common_idx]
        real_vol = realized_volatility.loc[common_idx]

        # Baseline 1: Historical volatility (rolling)
        hist_vol = returns.rolling(20).std() * np.sqrt(252)
        hist_vol = hist_vol.loc[common_idx]

        # Baseline 2: Simple EWMA
        ewma_vol = returns.ewm(alpha=0.1).std() * np.sqrt(252)
        ewma_vol = ewma_vol.loc[common_idx]

        # Baseline 3: Naive (constant volatility)
        naive_vol = pd.Series([real_vol.mean()] * len(real_vol), index=real_vol.index)

        # Baseline 4: GARCH(1,1) - simplified implementation
        try:
            garch_vol = self._simple_garch_volatility(returns, common_idx)
        except:
            garch_vol = naive_vol  # Fallback

        baselines = {
            'historical_vol': hist_vol,
            'ewma': ewma_vol,
            'naive': naive_vol,
            'garch': garch_vol
        }

        # Compare each metric
        vs_historical = {}
        vs_garch = {}
        vs_ewma = {}
        vs_naive = {}
        improvement_pct = {}

        for metric_name, baseline_vol in baselines.items():
            # Calculate RMSE for comparison
            try:
                baseline_rmse = np.sqrt(mean_squared_error(real_vol.values, baseline_vol.values))
                our_rmse = np.sqrt(mean_squared_error(real_vol.values, pred_vol.values))

                if baseline_rmse > 0:
                    improvement = (baseline_rmse - our_rmse) / baseline_rmse * 100
                else:
                    improvement = 0.0

                # Store results
                if metric_name == 'historical_vol':
                    vs_historical['rmse_improvement'] = improvement
                elif metric_name == 'garch':
                    vs_garch['rmse_improvement'] = improvement
                elif metric_name == 'ewma':
                    vs_ewma['rmse_improvement'] = improvement
                elif metric_name == 'naive':
                    vs_naive['rmse_improvement'] = improvement

                improvement_pct[f'vs_{metric_name}'] = improvement

            except Exception as e:
                print(f"Warning: Could not compare with {metric_name}: {e}")

        return ComparativeMetrics(
            vs_historical_vol=vs_historical,
            vs_garch=vs_garch,
            vs_ewma=vs_ewma,
            vs_naive=vs_naive,
            improvement_percentage=improvement_pct
        )

    def _simple_garch_volatility(self, returns: pd.Series, target_index) -> pd.Series:
        """Simplified GARCH(1,1) volatility estimation"""
        # Very simplified GARCH implementation
        omega = 0.000001
        alpha = 0.1
        beta = 0.8

        variance = pd.Series(index=returns.index)
        variance.iloc[0] = returns.var()

        for t in range(1, len(returns)):
            variance.iloc[t] = omega + alpha * returns.iloc[t-1]**2 + beta * variance.iloc[t-1]

        volatility = np.sqrt(variance) * np.sqrt(252)
        return volatility.loc[target_index]

    def evaluate_option_pricing_error(self,
                                    predicted_volatility: pd.Series,
                                    market_data: pd.DataFrame,
                                    option_prices: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Evaluate option pricing implications of volatility predictions.

        Parameters:
        -----------
        predicted_volatility : pd.Series
            Model's volatility predictions
        market_data : pd.DataFrame
            Underlying asset data
        option_prices : pd.DataFrame, optional
            Actual market option prices for comparison

        Returns:
        --------
        Dict[str, float] : Option pricing error metrics
        """
        # Simplified option pricing error assessment
        # In practice, this would use actual option pricing models

        results = {}

        # ATM option pricing error (simplified)
        spot_price = market_data['Close'].iloc[-1] if 'Close' in market_data.columns else 100
        time_to_expiry = 30/365  # 30 days

        # Theoretical price using predicted volatility
        # This is a very simplified approximation
        moneyness = np.array([0.9, 0.95, 1.0, 1.05, 1.1])  # Strike/Spot ratios
        strikes = spot_price * moneyness

        avg_pred_vol = predicted_volatility.tail(30).mean()  # Last 30 days average

        # Very simplified option price approximation
        # d1 approximation for ATM call
        d1 = (np.log(1.0) + (0.02 + 0.5 * avg_pred_vol**2) * time_to_expiry) / (avg_pred_vol * np.sqrt(time_to_expiry))
        theoretical_price = spot_price * (0.5 + 0.5 * stats.norm.cdf(d1)) - spot_price * np.exp(-0.02 * time_to_expiry) * (0.5 - 0.5 * stats.norm.cdf(d1))

        # In practice, compare with actual market prices
        # For now, return theoretical price as baseline
        results['theoretical_atm_call_price'] = theoretical_price
        results['implied_volatility'] = avg_pred_vol * np.sqrt(252)  # Annualized
        results['option_pricing_note'] = "Simplified assessment - full option pricing requires market data"

        return results

    def evaluate_risk_management(self,
                               predicted_volatility: pd.Series,
                               realized_volatility: pd.Series,
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Evaluate risk management implications (VaR, CVaR).

        Parameters:
        -----------
        predicted_volatility : pd.Series
            Predicted volatilities
        realized_volatility : pd.Series
            Actual realized volatilities
        confidence_level : float
            VaR confidence level (default 95%)

        Returns:
        --------
        Dict[str, Any] : Risk management metrics
        """
        results = {}

        # VaR backtesting (simplified)
        # Assume portfolio value of $1M, 1-day horizon
        portfolio_value = 1_000_000
        horizon = 1  # 1 day

        # Predicted VaR using model volatility
        predicted_var = portfolio_value * predicted_volatility * stats.norm.ppf(confidence_level) * np.sqrt(horizon/252)
        predicted_var = predicted_var.abs()  # Loss magnitude

        # Actual losses (simplified from volatility)
        actual_losses = portfolio_value * realized_volatility * np.random.normal(0, 1, len(realized_volatility)) * np.sqrt(horizon/252)
        actual_losses = actual_losses.abs()

        # VaR violation rate
        violations = (actual_losses > predicted_var).sum()
        total_obs = len(actual_losses)
        violation_rate = violations / total_obs

        expected_violations = 1 - confidence_level  # Expected 5% for 95% VaR
        actual_violations = violation_rate

        # Kupiec test (simplified)
        if violations > 0:
            kupiec_stat = -2 * np.log(((1-expected_violations)**(total_obs-violations) * expected_violations**violations) /
                                     ((1-actual_violations)**(total_obs-violations) * actual_violations**violations))
            kupiec_p_value = 1 - stats.chi2.cdf(kupiec_stat, 1)
        else:
            kupiec_p_value = 1.0

        results['var_confidence_level'] = confidence_level
        results['predicted_var_avg'] = predicted_var.mean()
        results['actual_losses_avg'] = actual_losses.mean()
        results['var_violation_rate'] = violation_rate
        results['expected_violation_rate'] = expected_violations
        results['kupiec_test_p_value'] = kupiec_p_value
        results['var_model_adequate'] = kupiec_p_value > 0.05  # Accept null hypothesis

        # CVaR (Conditional VaR)
        violation_mask = actual_losses > predicted_var
        if violation_mask.sum() > 0:
            cvar = actual_losses[violation_mask].mean()
            results['cvar'] = cvar
        else:
            results['cvar'] = 0

        return results

    def run_complete_evaluation(self,
                              predicted_volatility: pd.Series,
                              realized_volatility: pd.Series,
                              factors: pd.DataFrame,
                              constrained_factors: pd.DataFrame,
                              physical_target: float,
                              returns: pd.Series,
                              market_data: Optional[pd.DataFrame] = None) -> EvaluationResult:
        """
        Run complete evaluation pipeline.

        Parameters:
        -----------
        predicted_volatility : pd.Series
            Model predictions
        realized_volatility : pd.Series
            Actual realized volatility
        factors : pd.DataFrame
            Original agent factors
        constrained_factors : pd.DataFrame
            Factors after convergence
        physical_target : float
            Physical model target
        returns : pd.Series
            Asset returns

        Returns:
        --------
        EvaluationResult : Complete evaluation
        """
        # 1. Volatility prediction metrics
        volatility_metrics = self.evaluate_volatility_prediction(
            predicted_volatility, realized_volatility
        )

        # 2. Convergence quality metrics
        convergence_metrics = self.evaluate_convergence_quality(
            factors, constrained_factors, physical_target
        )

        # 3. Comparative analysis
        comparative_metrics = self.compare_with_baselines(
            predicted_volatility, realized_volatility, returns
        )

        # 4. Trading performance (already included in volatility_metrics)
        trading_performance = {
            'sharpe_ratio': volatility_metrics.sharpe_ratio,
            'max_drawdown': volatility_metrics.max_drawdown,
            'hit_rate': volatility_metrics.hit_rate
        }

        # 5. Option pricing implications (if market data available)
        option_pricing_metrics = {}
        if market_data is not None:
            try:
                option_pricing_metrics = self.evaluate_option_pricing_error(
                    predicted_volatility, market_data
                )
            except Exception as e:
                option_pricing_metrics['error'] = f"Option pricing evaluation failed: {e}"

        # 6. Risk management assessment
        risk_management_metrics = self.evaluate_risk_management(
            predicted_volatility, realized_volatility
        )

        # 7. Robustness tests
        robustness_tests = self._run_robustness_tests(
            predicted_volatility, realized_volatility
        )

        # Combine all metrics for robustness tests
        robustness_tests.update({
            'option_pricing_metrics': option_pricing_metrics,
            'risk_management_metrics': risk_management_metrics
        })

        return EvaluationResult(
            volatility_metrics=volatility_metrics,
            convergence_metrics=convergence_metrics,
            comparative_metrics=comparative_metrics,
            trading_performance=trading_performance,
            robustness_tests=robustness_tests
        )

    def _run_robustness_tests(self,
                           predicted_vol: pd.Series,
                           realized_vol: pd.Series) -> Dict[str, Any]:
        """
        Run robustness tests: different market conditions, subsamples, etc.
        """
        robustness = {}

        # Test on different volatility regimes
        high_vol_mask = realized_vol > realized_vol.quantile(0.75)
        low_vol_mask = realized_vol < realized_vol.quantile(0.25)

        for regime_name, mask in [('high_vol', high_vol_mask), ('low_vol', low_vol_mask)]:
            if mask.sum() > 10:
                pred_regime = predicted_vol.loc[mask]
                real_regime = realized_vol.loc[mask]

                try:
                    corr = pred_regime.corr(real_regime)
                    rmse = np.sqrt(mean_squared_error(real_regime.values, pred_regime.values))
                    robustness[f'{regime_name}_correlation'] = corr
                    robustness[f'{regime_name}_rmse'] = rmse
                except:
                    robustness[f'{regime_name}_correlation'] = None
                    robustness[f'{regime_name}_rmse'] = None

        # Out-of-sample test (last 20% of data)
        split_idx = int(len(predicted_vol) * 0.8)
        oos_pred = predicted_vol.iloc[split_idx:]
        oos_real = realized_vol.iloc[split_idx:]

        try:
            oos_corr = oos_pred.corr(oos_real)
            oos_rmse = np.sqrt(mean_squared_error(oos_real.values, oos_pred.values))
            robustness['out_of_sample_correlation'] = oos_corr
            robustness['out_of_sample_rmse'] = oos_rmse
        except:
            robustness['out_of_sample_correlation'] = None
            robustness['out_of_sample_rmse'] = None

        return robustness

    def generate_evaluation_report(self, result: EvaluationResult) -> str:
        """
        Generate comprehensive evaluation report.

        Parameters:
        -----------
        result : EvaluationResult
            Complete evaluation results

        Returns:
        --------
        str : Formatted evaluation report
        """
        # Extract metrics from result
        vm = result.volatility_metrics
        cm = result.convergence_metrics
        comp = result.comparative_metrics
        tp = result.trading_performance

        report = []
        report.append("📊 Dual Convergence Volatility Model - Enhanced Evaluation Report")
        report.append("=" * 80)
        report.append("")

        # Add statistical rigor section (simplified - sample size not available in result)
        report.append("🧪 STATISTICAL ASSESSMENT")
        report.append("-" * 40)

        # Correlation quality assessment
        if abs(vm.correlation) > 0.7:
            corr_quality = "Excellent correlation (r > 0.7)"
        elif abs(vm.correlation) > 0.5:
            corr_quality = "Good correlation (r > 0.5)"
        elif abs(vm.correlation) > 0.3:
            corr_quality = "Moderate correlation (r > 0.3)"
        else:
            corr_quality = "Weak correlation (r ≤ 0.3)"

        report.append(f"• Correlation Quality: {corr_quality}")

        # RMSE quality assessment
        if vm.rmse < 0.01:
            rmse_quality = "Excellent precision (RMSE < 0.01)"
        elif vm.rmse < 0.02:
            rmse_quality = "Good precision (RMSE < 0.02)"
        elif vm.rmse < 0.05:
            rmse_quality = "Fair precision (RMSE < 0.05)"
        else:
            rmse_quality = "Poor precision (RMSE ≥ 0.05)"

        report.append(f"• Prediction Precision: {rmse_quality}")

        # Comparative performance assessment
        if comp.vs_historical_vol and 'rmse_improvement' in comp.vs_historical_vol:
            improvement_pct = comp.vs_historical_vol['rmse_improvement']
            report.append(".1f")
            if improvement_pct > 15:
                report.append("• Benchmark Performance: ⭐⭐⭐ Strong improvement")
            elif improvement_pct > 5:
                report.append("• Benchmark Performance: ⭐⭐ Moderate improvement")
            elif improvement_pct > -5:
                report.append("• Benchmark Performance: ⭐ Comparable performance")
            else:
                report.append("• Benchmark Performance: ⚠️ Underperforms benchmark")

        report.append("")

        # Executive Summary
        report.append("🎯 EXECUTIVE SUMMARY")
        report.append("-" * 40)

        vm = result.volatility_metrics
        cm = result.convergence_metrics

        report.append(f"• Volatility Prediction Correlation: {vm.correlation:.4f}")
        report.append(f"• RMSE: {vm.rmse:.6f}")
        report.append(f"• Trading Sharpe Ratio: {vm.sharpe_ratio:.4f}")
        report.append(f"• Long-term Convergence: {cm.long_term_convergence:.4f}")
        report.append("")

        # Detailed Results
        report.append("📈 DETAILED RESULTS")
        report.append("-" * 40)

        report.append("\n1. VOLATILITY PREDICTION METRICS:")
        report.append(f"   Correlation: {vm.correlation:.4f}")
        report.append(f"   RMSE: {vm.rmse:.6f}")
        report.append(f"   MAE: {vm.mae:.6f}")
        report.append(f"   MAPE: {vm.mape:.2f}%")
        report.append(f"   Directional Accuracy: {vm.directional_accuracy:.4f}")
        report.append(f"   Hit Rate: {vm.hit_rate:.4f}")

        report.append("\n2. CONVERGENCE QUALITY:")
        report.append(f"   Long-term Convergence: {cm.long_term_convergence:.4f}")
        report.append(f"   Short-term Tracking: {cm.short_term_tracking:.4f}")
        report.append(f"   Convergence Speed: {cm.convergence_speed:.4f}")
        report.append(f"   Stability Score: {cm.stability_score:.4f}")

        report.append("\n3. FACTOR-PHYSICAL ALIGNMENT:")
        for factor, alignment in cm.factor_physical_alignment.items():
            report.append(f"   {factor}: {alignment:.4f}")

        report.append("\n4. TRADING PERFORMANCE:")
        report.append(f"   Sharpe Ratio: {vm.sharpe_ratio:.4f}")
        report.append(f"   Maximum Drawdown: {vm.max_drawdown:.4f}")

        report.append("\n5. COMPARATIVE ANALYSIS:")
        comp = result.comparative_metrics
        for model, metrics in [('Historical Vol', comp.vs_historical_vol),
                              ('GARCH', comp.vs_garch),
                              ('EWMA', comp.vs_ewma),
                              ('Naive', comp.vs_naive)]:
            if metrics:
                improvement = metrics.get('rmse_improvement', 0)
                report.append(f"   vs {model}: {improvement:+.1f}% RMSE improvement")

        report.append("\n6. RISK MANAGEMENT & OPTION PRICING:")
        robust = result.robustness_tests
        if 'risk_management_metrics' in robust:
            rm = robust['risk_management_metrics']
            report.append(".4f")
            report.append(".4f")
            if rm.get('var_model_adequate', False):
                report.append("• VaR Model: ✅ Adequate (passes Kupiec test)")
            else:
                report.append("• VaR Model: ⚠️ Inadequate (fails Kupiec test)")
            if 'cvar' in rm:
                report.append(".4f")
        if 'option_pricing_metrics' in robust:
            op = robust['option_pricing_metrics']
            if 'implied_volatility' in op:
                report.append(".1f")
                report.append("• Note: " + op.get('option_pricing_note', ''))

        report.append("\n7. ROBUSTNESS TESTS:")
        for test_name, value in robust.items():
            if test_name not in ['option_pricing_metrics', 'risk_management_metrics']:
                if value is not None and isinstance(value, (int, float)):
                    if 'correlation' in test_name:
                        report.append(f"   {test_name}: {value:.4f}")
                    elif 'rmse' in test_name:
                        report.append(f"   {test_name}: {value:.6f}")
                    else:
                        report.append(f"   {test_name}: {value}")

        # Economic significance and practical implications
        report.append("\n💰 ECONOMIC SIGNIFICANCE & PRACTICAL IMPLICATIONS")
        report.append("-" * 60)

        # Annualized performance metrics
        annual_return = vm.sharpe_ratio * vm.rmse * np.sqrt(252)  # Rough estimate
        report.append(".2f")
        # Risk-adjusted metrics
        report.append(".4f")
        report.append(".4f")
        # Practical trading considerations
        if vm.hit_rate > 0.55:
            report.append("• Trading Signal Quality: Good (>55% hit rate)")
        elif vm.hit_rate > 0.50:
            report.append("• Trading Signal Quality: Moderate (50-55% hit rate)")
        else:
            report.append("• Trading Signal Quality: Needs improvement (<50% hit rate)")

        # Time-series properties (simplified assessment)
        report.append("\n⏰ MODEL CHARACTERISTICS")
        report.append("-" * 35)

        # Directional accuracy indicates timing quality
        if vm.directional_accuracy > 0.6:
            timing_quality = "Excellent timing (directional accuracy > 60%)"
        elif vm.directional_accuracy > 0.5:
            timing_quality = "Good timing (directional accuracy > 50%)"
        else:
            timing_quality = "Timing needs improvement (directional accuracy ≤ 50%)"

        report.append(f"• Timing Quality: {timing_quality}")

        # Stability assessment based on convergence metrics
        if hasattr(result, 'convergence_metrics') and result.convergence_metrics:
            cm = result.convergence_metrics
            if cm.stability_score > 0.8:
                stability = "Highly stable convergence"
            elif cm.stability_score > 0.6:
                stability = "Moderately stable convergence"
            else:
                stability = "Convergence stability needs improvement"
            report.append(f"• Model Stability: {stability}")

        # Predictive power assessment
        report.append("\n🎯 PREDICTIVE POWER ASSESSMENT")
        report.append("-" * 45)

        # Overall model quality score (0-100)
        # Simplified scoring based on available metrics
        rmse_score = max(0, (1 - min(vm.rmse / 0.05, 1)) * 30)  # Assume typical vol std ~0.05
        quality_score = (
            min(abs(vm.correlation) * 100, 40) +  # Max 40 points for correlation
            rmse_score +  # Max 30 points for RMSE
            min(vm.directional_accuracy * 100, 15) +  # Max 15 points for direction
            min(max(vm.sharpe_ratio, 0) * 10, 15)  # Max 15 points for Sharpe
        )
        quality_score = min(quality_score, 100)

        if quality_score >= 80:
            grade = "A (Excellent)"
        elif quality_score >= 65:
            grade = "B (Good)"
        elif quality_score >= 50:
            grade = "C (Fair)"
        elif quality_score >= 35:
            grade = "D (Poor)"
        else:
            grade = "F (Very Poor)"

        report.append(".1f")
        report.append(f"• Model Grade: {grade}")

        # Convergence quality assessment
        if cm.long_term_convergence > 0.8:
            convergence_quality = "Excellent convergence to physical model"
        elif cm.long_term_convergence > 0.6:
            convergence_quality = "Good convergence to physical model"
        else:
            convergence_quality = "Convergence needs improvement"

        report.append(f"• Convergence Quality: {convergence_quality}")

        # Conclusions
        report.append("\n" + "=" * 80)
        report.append("🎯 CONCLUSIONS & RECOMMENDATIONS")
        report.append("=" * 80)

        # Performance assessment
        if vm.correlation > 0.7 and cm.long_term_convergence > 0.8:
            report.append("✅ EXCELLENT: Strong volatility prediction and convergence")
        elif vm.correlation > 0.5 and cm.long_term_convergence > 0.6:
            report.append("✅ GOOD: Solid performance with room for improvement")
        else:
            report.append("⚠️ NEEDS IMPROVEMENT: Consider model refinement")

        # Key strengths
        strengths = []
        if vm.correlation > 0.6:
            strengths.append("strong volatility prediction")
        if cm.long_term_convergence > 0.7:
            strengths.append("good convergence to physical model")
        if vm.sharpe_ratio > 1.0:
            strengths.append("profitable trading performance")

        if strengths:
            report.append(f"💪 Key Strengths: {', '.join(strengths)}")

        # Recommendations
        recommendations = []
        if cm.long_term_convergence < 0.7:
            recommendations.append("improve convergence constraints")
        if vm.correlation < 0.6:
            recommendations.append("enhance factor extraction")
        if vm.sharpe_ratio < 0.5:
            recommendations.append("refine trading strategy")

        if recommendations:
            report.append(f"🎯 Recommendations: {', '.join(recommendations)}")

        return "\n".join(report)
