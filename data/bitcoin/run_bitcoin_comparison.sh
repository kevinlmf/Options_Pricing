#!/bin/bash

# Bitcoin Options Model Comparison - Quick Start Script

echo "=========================================="
echo "Bitcoin Options Model Comparison Pipeline"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Install requirements
echo "Installing dependencies..."
pip3 install -q -r requirements_bitcoin.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "⚠ Warning: Some dependencies may have failed to install"
fi

echo ""
echo "=========================================="
echo "Running Model Comparison Pipeline"
echo "=========================================="
echo ""

# Run the pipeline
python3 bitcoin_model_comparison.py "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Pipeline completed successfully!"
    echo "=========================================="
    echo ""
    echo "📊 Results saved to: ./bitcoin_comparison_results/"
    echo ""
    echo "Key files:"
    echo "  - pnl_comparison.png              (P&L charts)"
    echo "  - risk_metrics.png                (Risk analysis)"
    echo "  - model_comparison_summary.csv    (Summary table)"
    echo "  - model_comparison_report.md      (Full report)"
    echo ""
else
    echo ""
    echo "❌ Pipeline failed. Check logs above for errors."
    exit 1
fi
