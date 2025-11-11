# Options Pricing System

## Core Purpose

**Finding the Optimal Measure that Best Explains Market Price Dynamics**

This project aims to discover a **measure-theoretic framework** that:

1. **Best Explains Market Price Movements**: Identifies the probability measure (P, Q, Q*) that most accurately captures real market dynamics
2. **Predicts Market Patterns**: Provides superior forecasting accuracy for volatility, drift, and regime changes
3. **Discovers Factors & Trends**: Automatically identifies market factors (volatility drivers, drift components, regime indicators) and tracks their evolution over time
4. **Balances Accuracy & Interpretability**: Combines predictive power with structural understanding—explains *why* prices move, not just *what* happens
5. **Converges to Risk-Neutral Pricing**: Ensures the discovered measure converges to risk-neutral (Q) pricing in the long run, maintaining no-arbitrage consistency

**Key Innovation**: Multi-agent structural modeling with adaptive learning, validated by Monte Carlo simulation, converging toward a unified measure that explains both real-world dynamics (P) and risk-neutral pricing (Q).

---

## System Overview

Integrated options trading system with portfolio optimization, risk management, and real-time monitoring, built around the core goal of measure discovery and convergence.

## System Architecture

```
Market Data → Forecasting (σ, μ) → Validation → Pricing → Optimization → Risk Control → Portfolio Construction → Monitoring
```

### 8-Layer Pipeline

| **Layer** | **Function** | **Output** |
|:----------|:-------------|:------------|
| **1. Data** | Market signals | Forecasted Prices / Volatility |
| **2. Forecasting** | Multi-Agent Simulator | σ, μ, Regime |
| **3. Validation** | Rust Monte Carlo (50k) | Validated Parameters + Confidence |
| **4. Pricing** | Black-Scholes, Heston, SABR | Option Prices, Greeks |
| **5. Optimization** | CVaR Optimizer | Optimal Positions |
| **6. Risk Control** | VaR, CVaR, Greek Limits | Approved Orders |
| **7. Portfolio Construction** | Trade execution | Executed Trades, Portfolio |
| **8. Monitoring** | Real-time tracking | Live Metrics, P&L |

## Project Structure

```
Options_Pricing/
├── time_series_forecasting/multi_agent/    # Multi-agent forecasting 
├── rust_monte_carlo/                       # Rust MC validation 
├── models/options_pricing/                 # Black-Scholes, Heston, SABR
├── models/optimization_methods/             # Portfolio optimizer
├── risk/                                  # Risk management 
└── examples/
    ├── multi_agent_vs_traditional_demo.py  # When to use what 
    ├── validated_integrated_demo.py        # Complete pipeline 
    └── validated_multi_agent_demo.py       # Detailed validation 
```
## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/kevinlmf/Options_Pricing
cd Options_Pricing

# Install Python dependencies pinned to the NumPy 1.26 ABI
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Build the Rust Monte Carlo accelerator (required for demos; 20-130x speedup)
# Install Rust first if needed: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
./rust_monte_carlo/build.sh
```

### Running Demos

```bash
chmod +x run_demo.sh
./run_demo.sh
```

> **Note:** Run `./rust_monte_carlo/build.sh` once per machine (and whenever the Rust code changes) before executing the demos so that the Monte Carlo validator is available.

## Key Features
**The Essence of Options Trading: Betting on Volatility**

Options trading is fundamentally about **betting on volatility** - traders are essentially wagering on whether the underlying asset will move enough (in either direction) to make the option profitable. This is why volatility is the most critical parameter in option pricing.

**What traders focus on:**
- **Volatility**  
- **Implied Volatility (IV)**: Market's expectation of future volatility
  - **Realized Volatility**: Actual volatility that occurs
  - **Volatility Trading**: Long volatility (buy options) vs. Short volatility (sell options)
- **Direction** ⭐⭐⭐ (Secondary - for directional trading)
  - Delta hedging, directional plays
  - Less important because options can profit from movement in either direction


### 1. Multi-Agent Structural Forecasting 

**Why Multi-Agent? Adaptive Learning & Factor Discovery**

Multi-agent systems continuously adjust agent behaviors to:
- **Better predict and learn from markets**: Agents adapt their strategies based on market feedback, improving prediction accuracy over time
- **Discover key factors automatically**: Market factors emerge naturally from agent interactions:
  - **Market Maker** → Discovers volatility factors from bid-ask spread dynamics
  - **Arbitrageur** → Discovers drift factors from arbitrage opportunities
  - **Noise Trader** → Discovers regime factors from trading patterns
- **No manual feature engineering**: Factors are discovered, not pre-specified


### 2. Rust-Accelerated Monte Carlo Validation 

**Performance:**

| Simulations | Python | Rust | Speedup |
|-------------|--------|------|---------|
| 50,000      | 6.0s   | 0.1s | **60x** |
| 100,000     | 12.5s  | 0.6s | **21x** |

**Result**: Real-time validation (~100ms) vs Python's 6-12 seconds.

### 3. Risk Management (`risk/` folder) 

- **VaR Models**: Historical, Parametric, Monte Carlo VaR
- **CVaR Models**: Conditional Value at Risk (Expected Shortfall)
- **Portfolio Risk**: Portfolio-level metrics, risk attribution
- **Option Risk**: Greeks-based analysis, option-specific VaR/CVaR
- **Integrated Risk**: Unified risk for stock-option portfolios

Used in: Portfolio optimization (CVaR constraints), pre-trade risk checks, real-time monitoring.

## Future Work

### Current Limitations

- **Monte Carlo Validation**: Multi-agent forecasts often fail validation (p-value < 0.05), indicating predictions deviate from simulated distributions
- **Drift Prediction**: While improved, drift prediction accuracy still needs enhancement (0.13% difference in best case)
- **Validation Pass Rate**: Currently 0% (by design - strict validation protects against unreliable forecasts)

### Planned Improvements

1. **Enhanced Agent Learning**
   - Add parameter update mechanisms based on prediction errors
   - Implement reinforcement learning framework (reward = prediction accuracy)
   - Experience replay for learning from historical interactions

2. **Improved Validation**
   - Relax validation criteria for regime-change scenarios
   - Adaptive validation thresholds based on market conditions
   - Ensemble validation combining multiple statistical tests

3. **Better Factor Discovery**
   - Dynamic factor weighting based on market regime
   - Multi-factor models combining agent-discovered factors
   - Factor interaction analysis

4. **Real-time Adaptation**
   - Online learning from streaming market data
   - Continuous parameter adjustment based on validation feedback
   - Adaptive agent behavior based on recent performance

---

MIT License | Explore interaction and uncertainty in both pricing and life, with hope even in the depth of winter☀️
