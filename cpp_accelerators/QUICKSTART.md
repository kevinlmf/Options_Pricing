# Quick Start - C++ Acceleration

## 快速构建(5分钟)

### Step 1: 安装依赖

```bash
pip install pybind11
```

**macOS额外步骤**:
```bash
brew install libomp
```

### Step 2: 构建C++模块

```bash
cd cpp_accelerators
make
```

你应该看到:
```
g++ -std=c++11 -O3 ... heston_cpp.so
g++ -std=c++11 -O3 ... sabr_cpp.so
```

### Step 3: 测试

```bash
python3 test_cpp_modules.py
```

看到 `All tests passed!` 就成功了!

### Step 4: 运行优化后的bitcoin对比

```bash
cd ..
python3 bitcoin/bitcoin_model_comparison.py
```

应该在30秒内完成!

## 一键脚本

```bash
cd cpp_accelerators
chmod +x build.sh
./build.sh
# 按照提示操作,脚本会自动构建和测试
```

## 验证加速效果

### Python实现(慢)

```python
from models.heston import HestonModel, HestonParameters
import time

params = HestonParameters(
    S0=100, K=105, T=0.25, r=0.05, q=0.02,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5
)
model = HestonModel(params)

# 强制使用Python
start = time.time()
price = model.option_price('call', use_cpp=False)
print(f"Python: {time.time()-start:.4f}s, Price: {price:.4f}")
```

### C++加速(快!)

```python
# 自动使用C++
start = time.time()
price = model.option_price('call', use_cpp=True)
print(f"C++: {time.time()-start:.4f}s, Price: {price:.4f}")
```

预期看到8倍速度提升!

## 故障排除

### 找不到libomp (macOS)

```bash
brew install libomp
export LDFLAGS="-L/usr/local/opt/libomp/lib"
export CPPFLAGS="-I/usr/local/opt/libomp/include"
make clean && make
```

### 找不到pybind11

```bash
pip install pybind11
python3 -c "import pybind11; print(pybind11.get_include())"
```

### 编译错误

尝试不使用OpenMP:
```bash
# 编辑Makefile,注释掉:
# OPENMP_FLAGS = -fopenmp
# OPENMP_LIBS = -lgomp
make clean && make
```

## 下一步

查看 `ACCELERATION_GUIDE.md` 了解:
- 详细性能分析
- 高级使用方式
- 批量定价技巧
- 进一步优化建议

查看 `README.md` 了解:
- API文档
- 技术细节
- 完整功能列表
