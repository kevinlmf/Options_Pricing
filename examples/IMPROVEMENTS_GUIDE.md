# Multi-Agent 改进指南

## 问题诊断

当前 Multi-Agent 系统预测的波动率经常达到 200%，导致验证失败。主要原因：

1. **参数系数过大**：`mean_spread * 20 + var_spread * 50` 放大了 spread 的影响
2. **缺乏市场状态感知**：没有根据历史波动率调整预测
3. **固定上限**：无论市场状态如何，都限制在 200%

## 改进方案

### 1. 使用改进版本的 Agent

已创建 `agents_improved.py`，包含以下改进：

#### 参数校准
```python
# 原始版本
implied_vol = 0.2 + mean_spread * 20 + var_spread * 50  # 系数过大

# 改进版本
implied_vol = 0.15 + mean_spread * 5 + var_spread * 10  # 更合理的系数
```

#### 市场状态感知
```python
# 根据历史波动率调整
historical_vol = np.std(recent_returns)
vol_ratio = realized_vol / (historical_vol + 1e-6)
spread = self.target_spread + realized_vol * 5 * vol_ratio
```

#### 自适应上限
```python
# 根据市场状态动态调整上限
if mean_spread < 0.01:      # 正常市场
    max_vol = 0.5   # 50%
elif mean_spread < 0.05:    # 波动市场
    max_vol = 1.0   # 100%
else:                        # 极端市场
    max_vol = 2.0   # 200%
```

### 2. 调整验证阈值

在创建 forecaster 时调整 p-value 阈值：

```python
from time_series_forecasting.multi_agent import create_validated_forecaster

# 更严格的验证（p < 0.01 才拒绝）
forecaster = create_validated_forecaster(
    validation_simulations=50_000,
    validation_threshold=0.01  # 更严格
)

# 更宽松的验证（p < 0.10 才拒绝）
forecaster = create_validated_forecaster(
    validation_simulations=50_000,
    validation_threshold=0.10  # 更宽松
)
```

### 3. 参数校准

改进版本包含自动校准功能：

```python
from time_series_forecasting.multi_agent.agents_improved import ImprovedMarketMaker

market_maker = ImprovedMarketMaker()
calibrated_params = market_maker.calibrate_from_historical(historical_prices)

# 使用校准后的参数
market_maker.vol_coefficient_mean = calibrated_params['vol_coefficient_mean']
market_maker.vol_coefficient_var = calibrated_params['vol_coefficient_var']
```

## 实施步骤

### 选项 1：直接修改现有代码（推荐用于生产）

修改 `agents.py` 中的 `MarketMaker.infer_parameter`：

```python
# 在 agents.py 中修改
implied_vol = 0.15 + mean_spread * 5 + var_spread * 10  # 使用改进的系数
return np.clip(implied_vol, 0.05, 1.0)  # 更合理的上限
```

### 选项 2：使用改进版本（推荐用于实验）

1. 修改 `agent_forecaster.py` 以支持使用改进的 agents
2. 或者直接替换 `MarketMaker` 类

### 选项 3：混合方案

保持现有代码不变，但在创建 forecaster 时使用改进的参数：

```python
# 创建 forecaster 时调整参数
forecaster = ValidatedMultiAgentForecaster(
    market_maker_risk_aversion=1.0,
    validation_threshold=0.05,  # 可调整
    # ... 其他参数
)
```

## 预期效果

实施改进后，预期：

- ✅ **波动率预测更合理**：从 200% 降低到 20-50%
- ✅ **验证通过率提高**：更多预测通过 Monte Carlo 验证
- ✅ **系统更可靠**：减少不必要的回退到 Traditional 方法

## 验证改进效果

运行演示并观察：

```bash
./run_demo.sh
```

检查：
1. Multi-Agent 预测的波动率是否更合理（< 100%）
2. 验证通过率是否提高
3. 系统是否更频繁地使用 Multi-Agent 预测

## 进一步优化

如果改进后仍有问题，可以考虑：

1. **更复杂的校准**：使用线性回归或机器学习方法校准系数
2. **多因子模型**：不仅考虑 spread，还考虑其他市场指标
3. **动态调整**：根据验证结果动态调整 agent 参数



