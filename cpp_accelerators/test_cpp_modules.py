#!/usr/bin/env python3
"""
Test script for C++ accelerated option pricing modules
"""

import numpy as np
import time
import sys

def test_heston():
    """Test Heston C++ module"""
    print("=" * 60)
    print("Testing Heston C++ Module")
    print("=" * 60)

    try:
        import heston_cpp
        print("✓ heston_cpp module loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import heston_cpp: {e}")
        print("  Run 'make' or 'python setup.py build_ext --inplace' first")
        return False

    # Test single option pricing
    print("\nTest 1: Single option pricing")
    try:
        result = heston_cpp.price_option(
            S0=100, K=105, T=0.25, r=0.05, q=0.02,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
            option_type='call'
        )
        print(f"  Call price: {result['price']:.6f}")
        print("  ✓ Single option pricing successful")
    except Exception as e:
        print(f"  ✗ Single option pricing failed: {e}")
        return False

    # Test batch pricing
    print("\nTest 2: Batch pricing")
    try:
        n = 50
        S0 = np.full(n, 100.0)
        K = np.linspace(80, 120, n)
        T = np.full(n, 0.25)
        r = np.full(n, 0.05)
        q = np.full(n, 0.02)
        v0 = np.full(n, 0.04)
        kappa = np.full(n, 2.0)
        theta = np.full(n, 0.04)
        xi = np.full(n, 0.3)
        rho = np.full(n, -0.5)
        is_call = np.full(n, True, dtype=bool)

        start = time.time()
        prices = heston_cpp.batch_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, is_call)
        elapsed = time.time() - start

        print(f"  Batch size: {n}")
        print(f"  Time: {elapsed*1000:.2f}ms")
        print(f"  Avg time per option: {elapsed*1000/n:.2f}ms")
        print(f"  Sample prices: {prices[:5]}")
        print("  ✓ Batch pricing successful")
    except Exception as e:
        print(f"  ✗ Batch pricing failed: {e}")
        return False

    return True


def test_sabr():
    """Test SABR C++ module"""
    print("\n" + "=" * 60)
    print("Testing SABR C++ Module")
    print("=" * 60)

    try:
        import sabr_cpp
        print("✓ sabr_cpp module loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import sabr_cpp: {e}")
        print("  Run 'make' or 'python setup.py build_ext --inplace' first")
        return False

    # Test single option pricing
    print("\nTest 1: Single option pricing")
    try:
        result = sabr_cpp.price_option(
            F=100, K=105, T=0.25, r=0.05,
            alpha=0.3, beta=0.7, rho=-0.3, nu=0.4,
            option_type='call'
        )
        print(f"  Call price: {result['price']:.6f}")
        print(f"  Implied vol: {result['implied_vol']:.6f}")
        print("  ✓ Single option pricing successful")
    except Exception as e:
        print(f"  ✗ Single option pricing failed: {e}")
        return False

    # Test batch pricing
    print("\nTest 2: Batch pricing")
    try:
        n = 100
        F = np.full(n, 100.0)
        K = np.linspace(80, 120, n)
        T = np.full(n, 0.25)
        r = np.full(n, 0.05)
        alpha = np.full(n, 0.3)
        beta = np.full(n, 0.7)
        rho = np.full(n, -0.3)
        nu = np.full(n, 0.4)
        is_call = np.full(n, True, dtype=bool)

        start = time.time()
        prices = sabr_cpp.batch_price(F, K, T, r, alpha, beta, rho, nu, is_call)
        elapsed = time.time() - start

        print(f"  Batch size: {n}")
        print(f"  Time: {elapsed*1000:.2f}ms")
        print(f"  Avg time per option: {elapsed*1000/n:.2f}ms")
        print(f"  Sample prices: {prices[:5]}")
        print("  ✓ Batch pricing successful")
    except Exception as e:
        print(f"  ✗ Batch pricing failed: {e}")
        return False

    # Test batch implied volatility
    print("\nTest 3: Batch implied volatility")
    try:
        start = time.time()
        ivs = sabr_cpp.batch_implied_vol(F, K, T, r, alpha, beta, rho, nu)
        elapsed = time.time() - start

        print(f"  Batch size: {n}")
        print(f"  Time: {elapsed*1000:.2f}ms")
        print(f"  Avg time per option: {elapsed*1000/n:.2f}ms")
        print(f"  Sample IVs: {ivs[:5]}")
        print("  ✓ Batch implied volatility successful")
    except Exception as e:
        print(f"  ✗ Batch implied volatility failed: {e}")
        return False

    return True


def benchmark():
    """Run performance benchmark"""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    try:
        import heston_cpp
        import sabr_cpp
    except ImportError:
        print("Skipping benchmark - modules not built")
        return

    # Heston benchmark
    print("\nHeston Model Benchmark:")
    for n in [10, 50, 100, 500]:
        S0 = np.full(n, 100.0)
        K = np.linspace(90, 110, n)
        T = np.full(n, 0.25)
        r = np.full(n, 0.05)
        q = np.full(n, 0.02)
        v0 = np.full(n, 0.04)
        kappa = np.full(n, 2.0)
        theta = np.full(n, 0.04)
        xi = np.full(n, 0.3)
        rho = np.full(n, -0.5)
        is_call = np.full(n, True, dtype=bool)

        start = time.time()
        prices = heston_cpp.batch_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, is_call)
        elapsed = time.time() - start

        print(f"  {n:4d} options: {elapsed*1000:7.2f}ms ({elapsed*1000/n:6.2f}ms per option)")

    # SABR benchmark
    print("\nSABR Model Benchmark:")
    for n in [10, 50, 100, 500, 1000]:
        F = np.full(n, 100.0)
        K = np.linspace(90, 110, n)
        T = np.full(n, 0.25)
        r = np.full(n, 0.05)
        alpha = np.full(n, 0.3)
        beta = np.full(n, 0.7)
        rho = np.full(n, -0.3)
        nu = np.full(n, 0.4)
        is_call = np.full(n, True, dtype=bool)

        start = time.time()
        prices = sabr_cpp.batch_price(F, K, T, r, alpha, beta, rho, nu, is_call)
        elapsed = time.time() - start

        print(f"  {n:4d} options: {elapsed*1000:7.2f}ms ({elapsed*1000/n:6.2f}ms per option)")


def main():
    """Main test runner"""
    print("C++ Accelerated Option Pricing - Test Suite")
    print()

    results = []

    # Test Heston
    results.append(("Heston", test_heston()))

    # Test SABR
    results.append(("SABR", test_sabr()))

    # Run benchmark
    benchmark()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")

    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
