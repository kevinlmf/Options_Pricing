#!/bin/bash
# Run Complete Integrated Options Trading Demo
# This demo shows the full pipeline including portfolio construction and trade execution

echo "=================================="
echo "Starting Options Pricing Demo..."
echo "=================================="
echo ""
echo "This demo demonstrates the complete trading pipeline:"
echo ""
echo "  1. Market Data & Forecasting"
echo "  2. Monte Carlo Validation"
echo "  3. Options Pricing"
echo "  4. Portfolio Optimization"
echo "  5. Risk Control"
echo "  6. Portfolio Construction & Trade Execution ⭐"
echo "  7. Real-time Monitoring"
echo ""
echo "=================================================================================="
echo ""

cd "$(dirname "$0")"
python3 examples/validated_integrated_demo.py

echo ""
echo "=================================="
echo "Demo completed!"
echo "=================================="
