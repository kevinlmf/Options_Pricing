"""
Real-time Monitoring Layer

Tracks:
1. Live P&L and returns
2. Portfolio Greeks in real-time
3. Risk metrics (VaR/CVaR)
4. Performance attribution
5. System health metrics

Provides WebSocket interface for dashboard visualization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MonitoringSnapshot:
    """Single monitoring snapshot"""
    timestamp: datetime

    # P&L
    portfolio_value: float
    cash: float
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float

    # Greeks
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_vega: float
    portfolio_theta: float

    # Risk
    var_95: float
    cvar_95: float
    current_drawdown: float
    max_drawdown: float

    # Performance
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0

    # System health
    forecast_latency_ms: float = 0.0
    pricing_latency_ms: float = 0.0
    risk_check_latency_ms: float = 0.0


class RealTimeMonitor:
    """
    Real-time monitoring system for the trading pipeline.

    Tracks all key metrics and provides:
    - Historical time series
    - Real-time alerts
    - Performance attribution
    - WebSocket streaming
    """

    def __init__(self,
                 initial_capital: float = 1000000.0,
                 history_window: int = 1000):
        """
        Initialize monitor.

        Parameters:
        -----------
        initial_capital : float
            Starting portfolio value
        history_window : int
            Number of snapshots to keep in memory
        """
        self.initial_capital = initial_capital
        self.history_window = history_window

        # State
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}

        # History (rolling window)
        self.snapshot_history: deque = deque(maxlen=history_window)
        self.pnl_history: deque = deque(maxlen=history_window)
        self.greeks_history: deque = deque(maxlen=history_window)

        # Peak tracking for drawdown
        self.peak_value = initial_capital
        self.max_drawdown = 0.0

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_pnls: List[float] = []

    def update(self,
              positions: Dict[str, float],
              option_prices: Dict[str, Dict],
              greeks: Dict[str, float],
              forecast_time: float = 0.0,
              pricing_time: float = 0.0,
              risk_check_time: float = 0.0) -> MonitoringSnapshot:
        """
        Update monitoring state with new data.

        Parameters:
        -----------
        positions : Dict[str, float]
            Current positions {option_id: quantity}
        option_prices : Dict[str, Dict]
            Current option prices {option_id: {'price': float, 'greeks': {...}}}
        greeks : Dict[str, float]
            Portfolio Greeks {'delta': ..., 'gamma': ..., etc.}
        forecast_time, pricing_time, risk_check_time : float
            Latencies in milliseconds

        Returns:
        --------
        MonitoringSnapshot
            Current monitoring state
        """
        timestamp = datetime.now()

        # Update positions
        self.positions = positions.copy()

        # Calculate portfolio value
        position_value = sum(
            qty * option_prices.get(opt_id, {}).get('price', 0.0)
            for opt_id, qty in positions.items()
        )
        self.portfolio_value = self.cash + position_value

        # P&L
        total_pnl = self.portfolio_value - self.initial_capital

        # Daily P&L (from yesterday's close)
        if len(self.snapshot_history) > 0:
            prev_value = self.snapshot_history[-1].portfolio_value
            daily_pnl = self.portfolio_value - prev_value
        else:
            daily_pnl = 0.0

        # Unrealized P&L
        unrealized_pnl = position_value

        # Drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Risk metrics (using historical returns)
        var_95, cvar_95 = self._calculate_risk_metrics()

        # Performance metrics
        sharpe, sortino, win_rate = self._calculate_performance_metrics()

        # Create snapshot
        snapshot = MonitoringSnapshot(
            timestamp=timestamp,
            portfolio_value=self.portfolio_value,
            cash=self.cash,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            unrealized_pnl=unrealized_pnl,
            portfolio_delta=greeks.get('delta', 0.0),
            portfolio_gamma=greeks.get('gamma', 0.0),
            portfolio_vega=greeks.get('vega', 0.0),
            portfolio_theta=greeks.get('theta', 0.0),
            var_95=var_95,
            cvar_95=cvar_95,
            current_drawdown=current_drawdown,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            win_rate=win_rate,
            forecast_latency_ms=forecast_time,
            pricing_latency_ms=pricing_time,
            risk_check_latency_ms=risk_check_time
        )

        # Store in history
        self.snapshot_history.append(snapshot)
        self.pnl_history.append(total_pnl)
        self.greeks_history.append(greeks)

        return snapshot

    def record_trade(self, pnl: float):
        """Record a completed trade"""
        self.total_trades += 1
        self.trade_pnls.append(pnl)

        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1

    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        if len(self.snapshot_history) == 0:
            return {}

        latest = self.snapshot_history[-1]

        return {
            'portfolio': {
                'value': latest.portfolio_value,
                'cash': latest.cash,
                'total_pnl': latest.total_pnl,
                'total_return_pct': (latest.total_pnl / self.initial_capital) * 100,
                'daily_pnl': latest.daily_pnl
            },
            'greeks': {
                'delta': latest.portfolio_delta,
                'gamma': latest.portfolio_gamma,
                'vega': latest.portfolio_vega,
                'theta': latest.portfolio_theta
            },
            'risk': {
                'var_95': latest.var_95,
                'cvar_95': latest.cvar_95,
                'current_drawdown_pct': latest.current_drawdown * 100,
                'max_drawdown_pct': latest.max_drawdown * 100
            },
            'performance': {
                'sharpe_ratio': latest.sharpe_ratio,
                'sortino_ratio': latest.sortino_ratio,
                'win_rate': latest.win_rate,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades
            },
            'system': {
                'forecast_latency_ms': latest.forecast_latency_ms,
                'pricing_latency_ms': latest.pricing_latency_ms,
                'risk_check_latency_ms': latest.risk_check_latency_ms,
                'total_latency_ms': (latest.forecast_latency_ms +
                                    latest.pricing_latency_ms +
                                    latest.risk_check_latency_ms)
            }
        }

    def get_time_series(self,
                       metric: str,
                       window: Optional[int] = None) -> Tuple[List[datetime], List[float]]:
        """
        Get time series for a specific metric.

        Parameters:
        -----------
        metric : str
            Metric name (e.g., 'portfolio_value', 'total_pnl', 'sharpe_ratio')
        window : Optional[int]
            Number of recent points (None = all)

        Returns:
        --------
        timestamps : List[datetime]
        values : List[float]
        """
        history = list(self.snapshot_history)
        if window is not None:
            history = history[-window:]

        timestamps = [snap.timestamp for snap in history]
        values = [getattr(snap, metric, 0.0) for snap in history]

        return timestamps, values

    def check_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []

        if len(self.snapshot_history) == 0:
            return alerts

        latest = self.snapshot_history[-1]

        # Drawdown alert
        if latest.current_drawdown > 0.10:  # 10% drawdown
            alerts.append({
                'level': 'WARNING' if latest.current_drawdown < 0.15 else 'CRITICAL',
                'type': 'DRAWDOWN',
                'message': f"Current drawdown: {latest.current_drawdown:.1%}",
                'value': latest.current_drawdown
            })

        # Greek limits alert
        if abs(latest.portfolio_delta) > 1000:
            alerts.append({
                'level': 'WARNING',
                'type': 'DELTA',
                'message': f"Portfolio delta exceeds limit: {latest.portfolio_delta:.0f}",
                'value': latest.portfolio_delta
            })

        if abs(latest.portfolio_vega) > 5000:
            alerts.append({
                'level': 'WARNING',
                'type': 'VEGA',
                'message': f"Portfolio vega exceeds limit: {latest.portfolio_vega:.0f}",
                'value': latest.portfolio_vega
            })

        # VaR alert
        if latest.var_95 > 100000:
            alerts.append({
                'level': 'WARNING',
                'type': 'VAR',
                'message': f"VaR exceeds $100k: ${latest.var_95:,.0f}",
                'value': latest.var_95
            })

        return alerts

    def _calculate_risk_metrics(self) -> Tuple[float, float]:
        """Calculate VaR and CVaR from historical returns"""
        if len(self.pnl_history) < 10:
            return 0.0, 0.0

        pnls = np.array(list(self.pnl_history))
        returns = np.diff(pnls) / self.initial_capital

        if len(returns) == 0:
            return 0.0, 0.0

        # VaR at 95% confidence
        var_95 = -np.percentile(returns, 5) * self.portfolio_value

        # CVaR (average of losses beyond VaR)
        var_cutoff = np.percentile(returns, 5)
        tail_losses = returns[returns <= var_cutoff]
        cvar_95 = -np.mean(tail_losses) * self.portfolio_value if len(tail_losses) > 0 else var_95

        return var_95, cvar_95

    def _calculate_performance_metrics(self) -> Tuple[float, float, float]:
        """Calculate Sharpe, Sortino, Win Rate"""
        if len(self.pnl_history) < 2:
            return 0.0, 0.0, 0.0

        pnls = np.array(list(self.pnl_history))
        returns = np.diff(pnls) / self.initial_capital

        if len(returns) == 0:
            return 0.0, 0.0, 0.0

        # Sharpe Ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0

        # Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
        sortino = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        # Win Rate
        win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0

        return sharpe, sortino, win_rate

    def generate_report(self) -> str:
        """Generate text report"""
        metrics = self.get_metrics_summary()

        if not metrics:
            return "No monitoring data available."

        report = []
        report.append("=" * 70)
        report.append(" REAL-TIME MONITORING REPORT")
        report.append("=" * 70)

        # Portfolio
        report.append("\n[PORTFOLIO]")
        report.append(f"  Value:        ${metrics['portfolio']['value']:,.2f}")
        report.append(f"  Cash:         ${metrics['portfolio']['cash']:,.2f}")
        report.append(f"  Total P&L:    ${metrics['portfolio']['total_pnl']:,.2f} ({metrics['portfolio']['total_return_pct']:+.2f}%)")
        report.append(f"  Daily P&L:    ${metrics['portfolio']['daily_pnl']:,.2f}")

        # Greeks
        report.append("\n[GREEKS]")
        report.append(f"  Delta:        {metrics['greeks']['delta']:,.2f}")
        report.append(f"  Gamma:        {metrics['greeks']['gamma']:,.2f}")
        report.append(f"  Vega:         {metrics['greeks']['vega']:,.2f}")
        report.append(f"  Theta:        {metrics['greeks']['theta']:,.2f}")

        # Risk
        report.append("\n[RISK]")
        report.append(f"  VaR (95%):    ${metrics['risk']['var_95']:,.2f}")
        report.append(f"  CVaR (95%):   ${metrics['risk']['cvar_95']:,.2f}")
        report.append(f"  Curr DD:      {metrics['risk']['current_drawdown_pct']:.2f}%")
        report.append(f"  Max DD:       {metrics['risk']['max_drawdown_pct']:.2f}%")

        # Performance
        report.append("\n[PERFORMANCE]")
        report.append(f"  Sharpe:       {metrics['performance']['sharpe_ratio']:.3f}")
        report.append(f"  Sortino:      {metrics['performance']['sortino_ratio']:.3f}")
        report.append(f"  Win Rate:     {metrics['performance']['win_rate']:.1%}")
        report.append(f"  Trades:       {metrics['performance']['total_trades']} (W: {metrics['performance']['winning_trades']}, L: {metrics['performance']['losing_trades']})")

        # System
        report.append("\n[SYSTEM HEALTH]")
        report.append(f"  Forecast:     {metrics['system']['forecast_latency_ms']:.2f}ms")
        report.append(f"  Pricing:      {metrics['system']['pricing_latency_ms']:.2f}ms")
        report.append(f"  Risk Check:   {metrics['system']['risk_check_latency_ms']:.2f}ms")
        report.append(f"  Total:        {metrics['system']['total_latency_ms']:.2f}ms")

        # Alerts
        alerts = self.check_alerts()
        if alerts:
            report.append("\n[ALERTS]")
            for alert in alerts:
                report.append(f"  [{alert['level']}] {alert['type']}: {alert['message']}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)
