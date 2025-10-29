# Options Pricing System

Integrated options trading system with portfolio optimization, risk management, and real-time monitoring.

## From Research to Production: End-to-End Trading System Lifecycle

## System Abstraction: Data → Environment → Execution → Monitoring

| **Stage** | **Core Function** | **Typical Components** | **Output** |
|:-----------|:------------------|:------------------------|:------------|
| **1. Data Layer** | Collect and model market signals | Market Data, LSTM, GARCH, Volatility Surface Estimator | Forecasted Prices / Volatility |
| **2. Environment Layer** | Simulate and value the trading world | Multi-Agent Simulator, Option Pricing Models (Black–Scholes / Heston / SABR), Market State Classifier | Market States, Theoretical Prices, Greeks |
| **3. Execution Layer** | Strategy generation and risk control | DP / RL Strategy Selector, Order Generator, Risk Controller (VaR, CVaR, Delta, Gamma, Vega, Theta, Concentration, Vol Regime) | Approved Orders, Portfolio Updates |
| **4. Monitoring Layer** | Real-time feedback and system optimization | Portfolio Tracker, WebSocket Dashboard, Metrics Logger | Live Risk/Return Visualization, System Logs |



## Architecture (7-Layer Pipeline)

```
Market Data → Forecasting (σ, μ) → Pricing (V_i) → Greeks (Δ, Γ, ν, Θ)
    → Portfolio Optimization (CVaR + Greek Constraints)
    → Risk Control → Execution → Real-time Monitoring
```

## Project Structure

```
Options_Pricing/
├── time_series_forecasting/       # Volatility forecasting
│   ├── classical_models/          # GARCH, ARIMA (Reduced-form)
│   ├── multi_agent/               # Multi-Agent (Structural) ⭐
│   └── forecast_comparator.py     # Comparison framework ⭐
├── models/
│   ├── options_pricing/           # Black-Scholes, Heston, SABR
│   └── optimization_methods/
│       └── strategy/portfolio_optimizer.py  # CVaR + Greek optimizer ⭐
├── risk/                          # VaR, CVaR, risk controls
├── evaluation_modules/
│   └── realtime_monitor.py        # Real-time monitoring ⭐
└── examples/
    └── integrated_portfolio_optimization_demo.py  # Complete pipeline ⭐
```

## Key Innovation: Reduced-form vs Structural Forecasting

| Approach | Nature | Best For |
|----------|--------|----------|
| **Time-Series** (LSTM/GARCH) | Reduced-form | Stable markets, efficiency |
| **Multi-Agent** (Simulation) | Structural | Regime changes, interpretability |

```python
# Compare forecasting methods
from time_series_forecasting.forecast_comparator import ForecastComparator
comparator = ForecastComparator()
comparison = comparator.compare_forecasts(prices, reduced_form, structural)
```

## Key Features

### 1. Portfolio Optimization
**Objective:** `maximize: α × E[Return] - β × CVaR_95 - γ × |Δ_p|`

Constraints: Greek limits (Delta, Vega, Gamma), delta neutrality, max position size

```python
from models.optimization_methods.strategy.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(risk_aversion=1.0, max_delta=500.0)
result = optimizer.optimize_portfolio(option_data, forecasted_returns)
print(f"Sharpe: {result.expected_sharpe}, CVaR: ${result.expected_cvar:,.2f}")
```

### 2. Real-Time Monitoring
Tracks P&L, Greeks (Δ, Γ, ν, Θ), VaR/CVaR, Sharpe/Sortino, drawdown, latency

```python
from evaluation_modules.realtime_monitor import RealTimeMonitor

monitor = RealTimeMonitor(initial_capital=1000000.0)
snapshot = monitor.update(positions, option_prices, greeks)
print(monitor.generate_report())
```

### 3. Pricing Models
Black-Scholes, Heston (stochastic vol), SABR, Binomial Tree, Local Volatility

```python
from models.options_pricing.black_scholes import BlackScholesModel, BSParameters
params = BSParameters(S0=50000, K=51000, T=30/365, r=0.05, sigma=0.6)
bs = BlackScholesModel(params)
price, delta = bs.call_price(), bs.delta('call')
```

### 4. Risk Management
Pre-trade checks: position limits, Greek limits, concentration, drawdown

```python
from risk.bitcoin_risk_controller import BitcoinRiskController
risk_controller = BitcoinRiskController(risk_limits, portfolio_value)
check = risk_controller.check_order(order)
```
### Setup Environment
```bash
git clone https://github.com/kevinlmf/Options_Pricing
cd Options_Pricing
```

## Quick Start

```bash
# One-click demo of all features (8 configurations)
./run_demo.sh

# Or run specific demo
python examples/integrated_portfolio_optimization_demo.py

# Or run specific demos
python3 examples/integrated_risk_trading_demo.py      # Multi-agent + Risk Control
python3 examples/complete_integration_demo.py         # Traditional + Options + Risk
python3 examples/final_pnl_comparison.py              # PnL comparison across all configs

# Start real-time API server
python3 api/realtime_monitor.py                       # Access at http://localhost:8000/docs
```

**Expected Output:**
```
[LAYER 1: MARKET DATA] ✓ Loaded 150 periods
[LAYER 2: FORECASTING] ✓ μ: 5.23%, σ: 62.45%
[LAYER 3: PRICING] ✓ Priced 24 options
[LAYER 4: OPTIMIZATION] ✓ Δ_p: 125.45, Sharpe: 1.234, CVaR: $45,230
[LAYER 5-7: RISK, EXECUTION, MONITORING] ✓ Complete!
```

**Installation**: Python 3.8+
```bash
pip install numpy scipy pandas matplotlib torch arch statsmodels fastapi uvicorn websockets
```




## Advanced Features

**Delta Hedging:**
```python
hedge = optimizer.hedge_to_delta_neutral(portfolio, options, ['spot', 'futures'])
```

**Strategy Selection:** Combine DP/RL selector with portfolio optimizer
```python
integrated = IntegratedStrategyOptimizer(dp_selector, optimizer)
strategy, positions = integrated.select_and_optimize(market_state, options, returns)
```

**Performance:** Pricing ~0.1ms, Optimization ~150ms, Total latency ~40ms

## Installation

```bash
pip install numpy scipy pandas scikit-learn torch gymnasium

# Optional C++ accelerators
cd cpp_accelerators && mkdir build && cd build && cmake .. && make
```

## Theory & References

See `theory/` for mathematical foundations (Black-Scholes PDE, Greeks, CVaR optimization, Heston calibration)

---

MIT License | Explore interaction and uncertainty in both pricing and life, with hope even in the depth of winter☀️
