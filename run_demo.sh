#!/bin/bash
# Run Integrated Portfolio Optimization Demo

echo "=================================="
echo "Starting Options Pricing Demo..."
echo "=================================="
echo ""

cd "$(dirname "$0")"
python examples/integrated_portfolio_optimization_demo.py

echo ""
echo "=================================="
echo "Demo completed!"
echo "=================================="
