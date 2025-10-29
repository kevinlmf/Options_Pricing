# Theoretical Framework: From Pure Mathematics to Financial Applications

## Overview

This document outlines the three-layer theoretical progression underlying derivatives risk management:

**Pure Math** → **Applied Math** → **Financial Models/Methods**

Each layer builds upon the previous, creating a robust foundation for quantitative finance.

---

## Layer 1: Pure Mathematics

*The fundamental mathematical language and symbolic framework*

### 1.1 Probability Theory
**Core Purpose**: Provides the language for randomness and uncertainty

- **Measure Theory**: (Ω, ℱ, P) - probability spaces
- **Random Variables**: Measurable functions X: Ω → ℝ
- **Distributions**: Normal, log-normal, exponential families
- **Moments**: E[X], Var(X), higher-order moments
- **Independence & Correlation**: Joint distributions, covariance

**Mathematical Foundation**:
```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
E[X] = ∫ x dF(x)
Var(X) = E[X²] - (E[X])²
```

### 1.2 Linear Algebra
**Core Purpose**: Multi-dimensional relationships and transformations

- **Vector Spaces**: ℝⁿ, inner products, norms
- **Matrices**: Linear transformations, eigenvalues/eigenvectors
- **Covariance Matrices**: Σ = E[(X - μ)(X - μ)ᵀ]
- **Matrix Decompositions**: Cholesky, SVD, eigendecomposition
- **Quadratic Forms**: xᵀAx for optimization and risk metrics

**Mathematical Foundation**:
```
Ax = λx (eigenvalue equation)
A = QΛQᵀ (spectral decomposition)
Σ = LLᵀ (Cholesky decomposition)
```

### 1.3 Calculus & Analysis
**Core Purpose**: Continuous change, derivatives, and integration

- **Differential Calculus**: ∂f/∂x, gradients ∇f, Hessians
- **Integral Calculus**: ∫f(x)dx, multiple integrals
- **Optimization**: Critical points, Lagrange multipliers
- **Functional Analysis**: Function spaces, convergence
- **Complex Analysis**: Analytic functions, Fourier transforms

**Mathematical Foundation**:
```
df/dx = lim[h→0] [f(x+h) - f(x)]/h
∫ᵃᵇ f(x)dx = F(b) - F(a)
∇f = 0 at critical points
```

---

## Layer 2: Applied Mathematics

*Mathematical tools directly applicable to modeling*

### 2.1 Stochastic Processes
**Built From**: Probability Theory + Calculus
**Purpose**: Mathematical language for asset price dynamics

- **Wiener Process**: W(t) with independent normal increments
- **Itô Calculus**: dX = μdt + σdW, Itô's lemma
- **Martingales**: E[X(t)|ℱₛ] = X(s) for s ≤ t
- **Markov Processes**: Memoryless property
- **Jump Processes**: Compound Poisson, Lévy processes

**Key Bridge to Finance**:
```
dS(t) = μS(t)dt + σS(t)dW(t) ← Asset price evolution
```

### 2.2 Partial Differential Equations (PDEs)
**Built From**: Calculus + Functional Analysis
**Purpose**: Framework for pricing equations

- **Parabolic PDEs**: ∂u/∂t + Lu = 0 (heat equation type)
- **Boundary Conditions**: Dirichlet, Neumann, Robin
- **Green's Functions**: Fundamental solutions
- **Variational Methods**: Weak formulations
- **Similarity Solutions**: Dimensional analysis

**Key Bridge to Finance**:
```
∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0 ← Black-Scholes PDE
```

### 2.3 Numerical Methods
**Built From**: Linear Algebra + Calculus
**Purpose**: Computational tools for complex problems

- **Finite Difference Methods**: Explicit/implicit schemes
- **Monte Carlo Methods**: Random sampling, variance reduction
- **Finite Element Methods**: Variational discretization
- **Optimization Algorithms**: Newton-Raphson, gradient descent
- **Fast Fourier Transform**: Spectral methods

**Key Bridge to Finance**:
```
Sᵢ₊₁ = Sᵢ exp((r - σ²/2)Δt + σ√Δt Zᵢ₊₁) ← MC simulation
```

### 2.4 Risk Measures & Optimization
**Built From**: Probability Theory + Optimization
**Purpose**: Quantifying and minimizing risk

- **Convex Analysis**: Convex functions, subdifferentials
- **Risk Functionals**: Coherent risk measures axioms
- **Portfolio Optimization**: Markowitz framework
- **Constraint Optimization**: KKT conditions
- **Robust Optimization**: Worst-case scenarios

**Key Bridge to Finance**:
```
VaRₐ = inf{x : P(L > x) ≤ 1-α} ← Risk quantification
```

### 2.5 Parameter Estimation & Calibration
**Built From**: Statistics + Optimization
**Purpose**: Fitting models to market data

- **Maximum Likelihood**: L(θ) = ∏f(xᵢ|θ)
- **Method of Moments**: Sample moments = theoretical moments
- **Bayesian Inference**: Prior + likelihood → posterior
- **Non-parametric Methods**: Kernel density estimation
- **Inverse Problems**: Parameter recovery from observations

**Key Bridge to Finance**:
```
σ̂ᵢᵥ = arg min Σ[V(σ) - Vₘₐᵣₖₑₜ]² ← Implied volatility calibration
```

---

## Layer 3: Financial Models & Methods

*Specific implementations in derivatives and risk management*

### 3.1 Black-Scholes Model
**Mathematical Foundation**:
- **Pure Math**: Normal distributions, exponential functions
- **Applied Math**: Geometric Brownian motion + Black-Scholes PDE
- **Numerical**: Analytical solution via error functions

**Implementation Chain**:
```
Probability Theory → Stochastic Process → PDE → Analytical Solution
     ↓                    ↓               ↓            ↓
Normal Distribution → dS = μSdt + σSdW → BS PDE → C = S₀Φ(d₁) - Ke⁻ʳᵀΦ(d₂)
```

### 3.2 Heston Stochastic Volatility Model
**Mathematical Foundation**:
- **Pure Math**: Complex analysis, characteristic functions
- **Applied Math**: 2D stochastic system + Fourier methods
- **Numerical**: FFT for option pricing

**Implementation Chain**:
```
Complex Analysis → Stochastic Processes → Characteristic Functions → FFT
      ↓                    ↓                       ↓                ↓
   Fourier Transform → 2D SDE System → φ(u,T) = e^(C+Dv₀+iulnS₀) → Option Prices
```

### 3.3 Value at Risk (VaR) & Conditional VaR
**Mathematical Foundation**:
- **Pure Math**: Quantile functions, tail distributions
- **Applied Math**: Risk measure theory + optimization
- **Numerical**: Monte Carlo simulation, historical methods

**Implementation Chain**:
```
Probability Theory → Risk Measures → Optimization → Portfolio VaR
      ↓                   ↓              ↓             ↓
  Tail Quantiles → Coherent Risk → Constraint Min → Risk Management
```

### 3.4 Monte Carlo Simulation
**Mathematical Foundation**:
- **Pure Math**: Law of large numbers, central limit theorem
- **Applied Math**: Stochastic processes + numerical integration
- **Numerical**: Random number generation, variance reduction

**Implementation Chain**:
```
Probability → Stochastic Process → Numerical Integration → Path Generation
     ↓              ↓                      ↓                    ↓
   LLN/CLT → Path Simulation → MC Estimator → E[f(X)] ≈ (1/N)Σf(Xᵢ)
```

### 3.5 Model Calibration & Implied Volatility
**Mathematical Foundation**:
- **Pure Math**: Inverse function theorem, optimization theory
- **Applied Math**: Parameter estimation + numerical optimization
- **Numerical**: Newton-Raphson, least squares fitting

**Implementation Chain**:
```
Optimization → Parameter Estimation → Numerical Methods → Market Fitting
     ↓                ↓                     ↓                  ↓
  Objective Min → Maximum Likelihood → Newton-Raphson → σᵢᵥ(K,T)
```

---

## Cross-Layer Dependencies

### Fundamental Mappings

| Pure Math | Applied Math | Financial Application |
|-----------|-------------|----------------------|
| Probability Theory | Stochastic Processes | Asset Price Models |
| Linear Algebra | Covariance Estimation | Portfolio Risk |
| Calculus | PDEs | Option Pricing |
| Complex Analysis | Fourier Methods | Characteristic Function Pricing |
| Optimization | Risk Measures | Risk Management |
| Statistics | Parameter Estimation | Model Calibration |

### Information Flow

```
Pure Mathematics
      ↓ (provides language & symbols)
Applied Mathematics
      ↓ (provides methods & tools)
Financial Models
      ↓ (provides specific applications)
Risk Management Systems
```

---

## Conclusion

This three-layer framework demonstrates how:

1. **Pure Mathematics** provides the foundational language and symbolic framework
2. **Applied Mathematics** develops these into practical modeling tools
3. **Financial Models** implement specific solutions for derivatives and risk management

Each layer is essential - without solid pure mathematics, applied methods lack rigor; without applied mathematics, financial models lack sophistication; without financial models, risk management lacks precision.

The beauty of this framework is that improvements at any layer cascade through the entire system, enabling continuous advancement in quantitative finance methodology.