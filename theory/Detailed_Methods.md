# Detailed Methods and Implementations

## Overview

This chapter derives pricing formulas from a unified perspective of risk and interest rates. Through measure transformation, we map market risk into martingale processes under the risk-neutral world, thereby deriving pricing PDEs. Under this framework, derivative prices are viewed as value functions satisfying specific PDEs (or expectation equations), while different models (e.g., Black-Scholes, Heston) merely correspond to different dynamical assumptions and measure structures.

---

## 🧩 I. Core Idea: Deriving Price from Risk

In derivative pricing, the existence of risk necessitates introducing a probability measure to define "fair price."
The measurement of risk, together with the interest rate (r), jointly determines the discounted future expected value.

In other words:

**"Price = Discounted Expectation under Some Measure"**

That is:
```
Vₜ = e^(-r(T-t)) E^Q[Payoff | ℱₜ]
```

---

## ⚙️ II. Logical Chain

| Level | Mathematical Object | Intuition | Corresponding Model |
|-------|---------------------|-----------|---------------------|
| **Stochastic Process** | dS = μSdt + σSdW | Randomness of market prices | GBM (Black-Scholes assumption) |
| **Measure Change** | From physical measure P to risk-neutral measure Q | Eliminate risk premium | Girsanov Theorem |
| **Value Function** | V(S,t) = E^Q[e^(-r(T-t))Π(Sₜ)] | Define derivative price | Risk-neutral pricing formula |
| **PDE / ODE** | ∂ₜV + ℒV - rV = 0 | Dynamic evolution | Black-Scholes PDE |
| **Boundary Conditions** | Payoff terminal condition | Value at expiry | European, Asian, American, etc. |
| **Numerical Methods** | FDM / MC / Fourier | Compute actual prices | Implementation & stability |

---

## 🧮 III. Why Do We Need a "New Measure"?

Under the original measure P, future returns contain a risk premium, making the value function unsuitable for directly computing "arbitrage-free prices."
We therefore introduce the **risk-neutral measure Q**, under which the discounted asset price becomes a **martingale**:

```
e^(-rt)Sₜ is a martingale under Q
```

This is the junction of "measurability + pricability":

- **Measurability** comes from the probability space (Ω, ℱ, Q)
- **Pricability** comes from the martingale property (i.e., no-arbitrage condition)

---

## 1. Black-Scholes Model: The Simplest Measure Change

### From Physical Measure to Risk-Neutral Measure

**Price dynamics under physical measure P**:
```
dS = μSdt + σSdW^P
```
where μ is the actual drift rate (includes risk premium)

**Price dynamics under risk-neutral measure Q**:
Through the Girsanov theorem, perform a measure change by setting:
```
W^Q = W^P + ((μ-r)/σ)t
```

Then under Q:
```
dS = rSdt + σSdW^Q
```

**Key Insight**: The risk premium (μ-r) is "absorbed" into the measure change; under the new measure, the price drifts only at the risk-free rate r.

### PDE Derivation of the Value Function

**Option value definition** (under Q):
```
V(S,t) = e^(-r(T-t)) E^Q[Payoff(Sₜ) | ℱₜ]
```

**Applying Itô's Lemma**:
Apply Itô's lemma to e^(-rt)V(S,t). Since it is a martingale under Q, its drift term must be zero:
```
∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
```

This is the famous **Black-Scholes PDE**.

### Analytical Solution: European Options

**Terminal condition** (boundary condition):
```
Call: V(S,T) = max(S-K, 0)
Put:  V(S,T) = max(K-S, 0)
```

**Analytical solutions**:

*European Call Option*:
```
C = S₀e^(-qT)Φ(d₁) - Ke^(-rT)Φ(d₂)
```

*European Put Option*:
```
P = Ke^(-rT)Φ(-d₂) - S₀e^(-qT)Φ(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
Φ(x) = standard normal cumulative distribution function
```

**Probabilistic Interpretation**:
- Φ(d₁) = "adjusted probability" that the underlying price exceeds the strike under Q
- Φ(d₂) = probability that the option finishes in-the-money under Q
- e^(-rT) = discount factor

### Greeks: Partial Derivatives of the Value Function

Greeks are sensitivities of the value function to various parameters, essentially partial derivatives of the PDE solution:

**Delta (Δ)** - First derivative w.r.t. underlying price:
```
Call: Δ = e^(-qT)Φ(d₁)
Put:  Δ = e^(-qT)[Φ(d₁) - 1]
```
Interpretation: Hedge ratio; holding Δ units of the underlying hedges the instantaneous price risk of the option.

**Gamma (Γ)** - Second derivative w.r.t. underlying price:
```
Γ = e^(-qT)φ(d₁)/(S₀σ√T)
```
Interpretation: Rate of change of Delta; measures "convexity risk" of the hedged portfolio.

**Vega (ν)** - Derivative w.r.t. volatility:
```
ν = S₀e^(-qT)φ(d₁)√T
```
Interpretation: Volatility risk exposure.

**Theta (Θ)** - Time decay:
```
Call: Θ = -S₀e^(-qT)φ(d₁)σ/(2√T) - rKe^(-rT)Φ(d₂) + qS₀e^(-qT)Φ(d₁)
Put:  Θ = -S₀e^(-qT)φ(d₁)σ/(2√T) + rKe^(-rT)Φ(-d₂) - qS₀e^(-qT)Φ(-d₁)
```
Interpretation: Rate of time value decay.

**Rho (ρ)** - Derivative w.r.t. risk-free rate:
```
Call: ρ = KTe^(-rT)Φ(d₂)
Put:  ρ = -KTe^(-rT)Φ(-d₂)
```
Interpretation: Interest rate risk exposure.

### Implementation Considerations

**Numerical Stability**:
- Extreme moneyness cases: S/K → 0 or S/K → ∞
- Asymptotic expansions for very short or very long time to expiry
- Robust normal CDF/PDF calculations

**Implied Volatility Calculation**:
Newton-Raphson iteration:
```
σₙ₊₁ = σₙ - [BS_Price(σₙ) - Market_Price] / Vega(σₙ)
```

**Extensions**:
- American options via binomial trees or finite difference methods
- Discrete dividend adjustments
- Currency options with foreign interest rate

---

## 2. Heston Model: Stochastic Volatility and Measure Change

### Why Stochastic Volatility?

Black-Scholes assumes constant volatility, but markets exhibit:
- **Volatility smile/skew**: Different implied volatilities for different strikes
- **Volatility clustering**: High and low volatility periods cluster
- **Leverage effect**: Volatility rises when stock price falls (negative correlation)

### Two-Factor System

**Under risk-neutral measure Q**:
```
dS(t) = rS(t)dt + √v(t)S(t)dW₁^Q(t)
dv(t) = κ(θ - v(t))dt + ξ√v(t)dW₂^Q(t)
```

Where:
```
dW₁^Q(t)dW₂^Q(t) = ρdt
```

**Parameter Interpretation** (from risk perspective):
- **v(t)**: Instantaneous variance (time-varying risk measure)
- **κ**: Mean reversion speed of variance (speed of risk "self-correction")
- **θ**: Long-run variance level (long-term average risk)
- **ξ**: Volatility of volatility ("risk of risk")
- **ρ**: Correlation between price and volatility (leverage effect, typically ρ < 0)

**Feller Condition**:
```
2κθ > ξ²
```
Ensures the variance process v(t) stays strictly positive (risk cannot be negative).

### Understanding Pricing from a Measure Perspective

**Value function** (under Q):
```
V(S,v,t) = e^(-r(T-t)) E^Q[Payoff(Sₜ) | Sₜ=S, vₜ=v]
```

**Corresponding PDE** (2D Feynman-Kac):
```
∂V/∂t + (1/2)vS²∂²V/∂S² + ρξvS∂²V/(∂S∂v) + (1/2)ξ²v∂²V/∂v²
    + rS∂V/∂S + κ(θ-v)∂V/∂v - rV = 0
```

**Interpretation**:
- First line: Diffusion terms (risk terms)
- Second line: Drift terms (under Q, all are r or mean-reverting)
- Last term: Discounting

### Characteristic Function Method

**Heston Characteristic Function**:
```
φ(u,T) = exp(C(u,T) + D(u,T)v₀ + iu ln(S₀))
```

**Component Functions**:
```
d = √[(ρξui - κ)² + ξ²(ui + u²)]
g = (κ - ρξui - d)/(κ - ρξui + d)

C(u,T) = rTui + (κθ/ξ²)[(κ - ρξui - d)T - 2ln((1 - ge^(-dT))/(1 - g))]
D(u,T) = ((κ - ρξui - d)/ξ²) × ((1 - e^(-dT))/(1 - ge^(-dT)))
```

### Fourier Inversion Pricing

**Call Option Formula**:
```
C = S₀e^(-qT)P₁ - Ke^(-rT)P₂
```

**Probability Integrals**:
```
Pⱼ = (1/2) + (1/π) ∫₀^∞ Re[e^(-iu ln K)φⱼ(u,T)/(iu)] du
```

where φ₁ and φ₂ are modified characteristic functions.

**Interpretation**:
- P₁, P₂ are analogous to Φ(d₁), Φ(d₂) in Black-Scholes
- But now obtained via Fourier inversion from characteristic functions
- Still essentially "adjusted probabilities" under Q

### Numerical Implementation

**FFT-Based Pricing**:
1. Discretize integration domain: u ∈ [0, U] with N points
2. Apply FFT to compute option prices for multiple strikes simultaneously
3. Use damping parameter α to ensure convergence

**Monte Carlo Simulation**:
Euler discretization (Full Truncation scheme):
```
Sᵢ₊₁ = Sᵢ exp((r - q - vᵢ/2)Δt + √(vᵢΔt)Z₁,ᵢ₊₁)
vᵢ₊₁ = vᵢ + κ(θ - vᵢ)Δt + ξ√(max(vᵢ,0)Δt)Z₂,ᵢ₊₁
```

Correlated random number generation:
```
Z₂ = ρZ₁ + √(1-ρ²)Z̃₂
```

### Model Calibration

**Objective Function** (minimize relative errors):
```
f(Θ) = Σᵢ [(V_market^i - V_model^i(Θ))/V_market^i]²
```

**Parameter Constraints** (ensure measure validity):
- v₀ > 0: Initial variance is positive
- κ > 0: Positive mean reversion
- θ > 0: Long-run variance is positive
- ξ > 0: Vol-of-vol is positive
- |ρ| < 1: Correlation bounds
- 2κθ > ξ²: Feller condition

**Optimization Methods**:
- Global optimization: Differential Evolution, Particle Swarm
- Local refinement: L-BFGS-B with parameter bounds
- Multi-start strategy to avoid local optima

---

## 3. Value at Risk (VaR): Understanding Risk Measures from a Measure Perspective

### Mathematical Definition

**Value at Risk** (under measure P):
```
VaRₐ(X) = inf{x ∈ ℝ : P(X ≤ x) ≥ α}
```

**Conditional Value at Risk (Expected Shortfall)**:
```
CVaRₐ(X) = E[X | X ≤ VaRₐ(X)]
```

**Interpretation**:
- VaR: Maximum loss at confidence level (1-α)
- CVaR: Conditional expected loss beyond VaR
- Note: These are measured under the **physical measure P**, not the risk-neutral measure Q

### Calculation Methods

**1. Historical Simulation**:
```
VaR₅% = 5th percentile of historical P&L distribution
```

Steps:
1. Collect historical asset returns: r₁, r₂, ..., rₙ
2. Apply current portfolio weights: P&Lᵢ = wᵀrᵢ
3. Sort P&L values: P&L₍₁₎ ≤ P&L₍₂₎ ≤ ... ≤ P&L₍ₙ₎
4. VaRₐ = P&L₍⌊αn⌋₎

**2. Parametric Method (Variance-Covariance)**:
Assume normal distribution: P&L ~ N(μ, σ²)
```
VaRₐ = μ + σΦ⁻¹(α)
```

For portfolios: σ² = wᵀΣw, where Σ is the covariance matrix

**3. Monte Carlo Simulation**:
```
1. Simulate asset price paths: S₁, S₂, ..., Sₙ
2. Calculate portfolio values: V₁, V₂, ..., Vₙ
3. Compute P&L: P&Lᵢ = Vᵢ - V₀
4. VaRₐ = α-th quantile of {P&L₁, ..., P&Lₙ}
```

### Coherent Risk Measures

A risk measure ρ is **coherent** if and only if it satisfies:

**1. Monotonicity**: X ≤ Y ⟹ ρ(X) ≥ ρ(Y)

**2. Translation Invariance**: ρ(X + c) = ρ(X) - c

**3. Positive Homogeneity**: ρ(λX) = λρ(X) for λ > 0

**4. Sub-additivity**: ρ(X + Y) ≤ ρ(X) + ρ(Y)

**Important Conclusions**:
- VaR **fails** sub-additivity (may penalize diversification)
- CVaR **is coherent** (encourages diversification)

### Backtesting and Validation

**Kupiec Test** (VaR model validation):
```
H₀: Violation rate = α
Test statistic: LR = -2ln[L(p̂)/L(α)]
where p̂ = number of violations / number of observations
```

**Stress Testing**:
- Extreme market scenario analysis
- Factor shock tests
- Historical scenario replications

---

## 4. Monte Carlo Methods: Simulating Expectations under a Measure

### Basic Theory

**Law of Large Numbers**:
```
(1/N)Σᵢ₌₁ᴺ f(Xᵢ) → E[f(X)] as N → ∞
```

**Central Limit Theorem**:
```
√N[(1/N)Σf(Xᵢ) - E[f(X)]] → N(0, Var[f(X)])
```

**Convergence Rate**: O(1/√N) - independent of dimension

**From a Measure Perspective**:
Monte Carlo essentially approximates the theoretical measure with the empirical measure:
```
μₙ = (1/N)Σᵢ₌₁ᴺ δ_Xᵢ → μ (weak convergence)
```

### Path Generation Methods

**1. Euler-Maruyama Scheme**:
For SDE: dX = μ(X,t)dt + σ(X,t)dW
```
Xₜ₊Δₜ = Xₜ + μ(Xₜ,t)Δt + σ(Xₜ,t)√Δt Z
```

**2. Milstein Scheme** (higher order):
```
Xₜ₊Δₜ = Xₜ + μΔt + σ√Δt Z + (1/2)σσ'Δt(Z² - 1)
```

**3. Exact Simulation** (when available):
For geometric Brownian motion:
```
S(T) = S₀ exp((r - σ²/2)T + σ√T Z)
```

### Variance Reduction Techniques

**1. Antithetic Variates**:
```
Use both Z and -Z in simulation
Estimator: [f(Z) + f(-Z)]/2
Variance reduction when f is monotonic
```

**2. Control Variates**:
```
f̃ = f(X) - β[g(X) - E[g(X)]]
Choose β to minimize Var[f̃]
Optimal: β* = Cov[f(X),g(X)]/Var[g(X)]
```

**3. Importance Sampling**:
```
E[f(X)] = ∫ f(x)p(x)dx = ∫ f(x)[p(x)/q(x)]q(x)dx
Estimator: (1/N)Σf(Yᵢ)[p(Yᵢ)/q(Yᵢ)] where Yᵢ ~ q
```

**Measure Perspective Interpretation**:
- Importance sampling = measure change (from q to p)
- Radon-Nikodym derivative = p(x)/q(x)

**4. Stratified Sampling**:
Divide [0,1] into k strata, sample within each:
```
Estimator: Σₖ(nₖ/N)X̄ₖ where nₖ = stratum k sample size
```

### Multi-Dimensional Simulation

**Cholesky Decomposition** (for correlated normals):
```
If Σ = LLᵀ (Cholesky decomposition)
Then X = μ + LZ gives X ~ N(μ, Σ)
```

**Principal Component Analysis**:
```
Σ = QΛQᵀ (eigendecomposition)
Generate Y ~ N(0, Λ), set X = μ + QY
```

### Quasi-Monte Carlo

**Low-Discrepancy Sequences**:
- Sobol sequences
- Halton sequences
- Latin hypercube sampling

**Convergence Rate**: O((log N)ᵈ/N) where d = dimension

Better than standard MC for smooth integrands in moderate dimensions.

---

## 5. PDE Numerical Methods: Directly Solving the Value Function

### Finite Difference Methods

**Grid Setup**:
```
Space: S ∈ [0, Sₘₐₓ] with ΔS = Sₘₐₓ/M
Time: t ∈ [0, T] with Δt = T/N
Grid points: (iΔS, jΔt) for i = 0,...,M; j = 0,...,N
```

**Discrete Operators**:
```
∂V/∂S ≈ (Vᵢ₊₁ʲ - Vᵢ₋₁ʲ)/(2ΔS)  (central difference)
∂²V/∂S² ≈ (Vᵢ₊₁ʲ - 2Vᵢʲ + Vᵢ₋₁ʲ)/(ΔS)²  (second difference)
∂V/∂t ≈ (Vᵢʲ⁺¹ - Vᵢʲ)/Δt  (forward difference)
```

**Explicit Scheme** (Forward Euler):
```
Vᵢʲ⁺¹ = Vᵢʲ + Δt[½σ²i²ΔS²(Vᵢ₊₁ʲ - 2Vᵢʲ + Vᵢ₋₁ʲ)/(ΔS)² +
         riΔS(Vᵢ₊₁ʲ - Vᵢ₋₁ʲ)/(2ΔS) - rVᵢʲ]
```

Stability condition: Δt ≤ (ΔS)²/(σ²Sₜₒₚ²)

**Implicit Scheme** (Backward Euler):
```
Vᵢʲ = Vᵢʲ⁺¹ + Δt[ℒVᵢʲ⁺¹]
```
Requires solving a tridiagonal system, but unconditionally stable.

**Crank-Nicolson Scheme**:
```
Vᵢʲ⁺¹ - Vᵢʲ = (Δt/2)[ℒVᵢʲ + ℒVᵢʲ⁺¹]
```
Second-order accurate in time, unconditionally stable.

### Boundary Conditions

**Far-field Boundaries**:
- S → 0: V ≈ discounted intrinsic value
- S → ∞: V ≈ S - Ke^(-r(T-t)) (call); V ≈ 0 (put)

**Free Boundary Problems** (American options):
Use penalty methods or linear complementarity formulation.

---

## 6. Model Calibration and Parameter Estimation

### Maximum Likelihood Estimation

**Log-Likelihood for Geometric Brownian Motion**:
Given observations S₀, S₁, ..., Sₙ at times 0, Δt, 2Δt, ..., nΔt

```
ℓ(μ, σ) = -½n ln(2πσ²Δt) - (1/(2σ²Δt))Σᵢ₌₁ⁿ[ln(Sᵢ/Sᵢ₋₁) - (μ - σ²/2)Δt]²
```

**MLE Solutions**:
```
μ̂ = (1/(nΔt))ln(Sₙ/S₀) + σ̂²/2
σ̂² = (1/(nΔt))Σᵢ₌₁ⁿ[ln(Sᵢ/Sᵢ₋₁) - μ̂Δt + σ̂²Δt/2]²
```

### Method of Moments

Match sample moments to theoretical moments:
```
Sample mean = Theoretical mean
Sample variance = Theoretical variance
...
```

For GBM with observations rᵢ = ln(Sᵢ/Sᵢ₋₁):
```
μ̂ = r̄/Δt + σ̂²/2
σ̂² = s²ᵣ/Δt
```

### Implied Parameter Calibration

**Objective Function**:
```
min Σᵢ wᵢ[V_model(Kᵢ, Tᵢ; θ) - V_market(Kᵢ, Tᵢ)]²
```

Common weights: wᵢ = 1/V_market(Kᵢ, Tᵢ) (relative error)

**Optimization Challenges**:
- Non-convex objective function
- Multiple local minima
- Parameter constraints
- Ill-conditioning near parameter boundaries

**Regularization Techniques**:
```
Objective = Fit Error + λ × Penalty(θ)
```
Where Penalty(θ) prevents extreme parameter values.

### Implied Volatility Surface

**Smile/Skew Phenomena**:
- At-the-money: Benchmark volatility
- Out-of-the-money puts: Higher implied vol (crash premium)
- Out-of-the-money calls: Varying patterns across markets

**Parametric Models**:

1. **SVI (Stochastic Volatility Inspired)**:
   ```
   σ²ᵢᵥ(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]
   ```

2. **SABR Model**:
   ```
   σᵢᵥ(K,T) ≈ (α/f̃^(1-β))[1 + ((1-β)²α²)/(24f̃^(2-2β))T + ...]
   ```

**Arbitrage-Free Conditions**:
- Calendar arbitrage: ∂σᵢᵥ/∂T ≥ constraints
- Butterfly arbitrage: ∂²C/∂K² ≥ 0

---

## Implementation Best Practices

### Code Structure
1. **Modular Design**: Separate mathematical components
2. **Parameter Validation**: Check bounds and conditions
3. **Numerical Stability**: Handle edge cases gracefully
4. **Performance Optimization**: Vectorization, caching
5. **Testing**: Unit tests for each mathematical component

### Common Pitfalls
1. **Numerical Overflow**: Use log-space calculations when possible
2. **PDF Underflow**: Implement robust normal PDF/CDF
3. **Parameter Bounds**: Enforce constraints during optimization
4. **Convergence Criteria**: Set appropriate tolerances
5. **Market Data Quality**: Clean and validate inputs

### Validation Methods
1. **Analytical Benchmarks**: Compare to known solutions
2. **Convergence Testing**: Verify O(h²) or O(1/√N) rates
3. **Cross-Validation**: Out-of-sample testing
4. **Stress Testing**: Extreme parameter regimes
5. **Market Comparison**: Validate against observed prices

---

## Summary: Unified Measure-PDE-Pricing Framework

This document presents a unified perspective on derivative pricing:

1. **Starting Point**: Randomness of market prices (stochastic processes)
2. **Core Transformation**: Measure change (P → Q) eliminates risk premium
3. **Value Definition**: V = E^Q[discounted Payoff] (risk-neutral pricing)
4. **Dynamic Evolution**: Characterized by PDEs or expectation equations
5. **Numerical Solution**: FDM, MC, Fourier methods, etc.
6. **Practical Calibration**: Back out model parameters from market data

All pricing models (Black-Scholes, Heston, Local Vol, etc.) follow this logical chain; they differ only in their specific assumptions about price dynamics.
