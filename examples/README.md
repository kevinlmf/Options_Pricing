# Examples Guide

## Core Examples (3 Essential Demos)

### 1. `multi_agent_vs_traditional_demo.py` ⭐ **MAIN DEMO**
**Purpose**: Shows when to use Multi-Agent vs Traditional methods

**Scenarios**:
- **Scenario 1**: Stable Market → Traditional performs better
- **Scenario 2**: Regime Change → Multi-Agent performs better  
- **Scenario 3**: Validation decides which method to use

**Key Insights**:
- Traditional methods excel in stable, well-behaved markets
- Multi-Agent methods adapt better to regime changes
- Validation ensures we use the right method automatically

**Run**: `./run_demo.sh` or `python3 examples/multi_agent_vs_traditional_demo.py`

---

### 2. `validated_integrated_demo.py` ⭐
**Purpose**: Complete pipeline with validation decision

**Shows**:
- Multi-Agent forecasting
- Monte Carlo validation (50k simulations)
- Auto-decision: Use Multi-Agent if validated, else fallback
- Complete portfolio optimization pipeline
- Risk control and monitoring

**Run**: `python3 examples/validated_integrated_demo.py`

---

### 3. `validated_multi_agent_demo.py` ⭐
**Purpose**: Detailed validation demonstration

**Shows**:
- Multi-agent structural forecasting
- Rust-accelerated Monte Carlo validation
- Option pricing with validated parameters
- Risk-adjusted position sizing
- Performance benchmarks (Rust vs Python)

**Run**: `python3 examples/validated_multi_agent_demo.py`

---

## Quick Start

```bash
# Main demo (recommended) - Shows when to use what
./run_demo.sh

# Or run specific demos
python3 examples/multi_agent_vs_traditional_demo.py      # When to use what
python3 examples/validated_integrated_demo.py            # Complete pipeline
python3 examples/validated_multi_agent_demo.py            # Detailed validation
```

## Demo Comparison

| Demo | Focus | Best For |
|------|-------|----------|
| `multi_agent_vs_traditional_demo.py` | When to use what | Understanding method selection |
| `validated_integrated_demo.py` | Complete pipeline | End-to-end system demonstration |
| `validated_multi_agent_demo.py` | Validation details | Deep dive into validation process |

