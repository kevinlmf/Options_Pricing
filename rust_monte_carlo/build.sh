#!/bin/bash
#
# Build script for Rust Monte Carlo module
#

set -e

# Ensure the script runs from the rust_monte_carlo directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔨 Building Rust Monte Carlo Validator..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not found. Please install from https://rustup.rs/"
    exit 1
fi

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "📦 Installing maturin..."
    pip install maturin
fi

# Build release version
echo "🚀 Building release version..."
maturin build --release

# Or develop mode (faster, for development)
if [ "$1" == "dev" ]; then
    echo "🔧 Building in development mode..."
    maturin develop
fi

# Copy to python directory
echo "📁 Copying to python directory..."
find target/wheels -name "*.whl" -exec pip install --force-reinstall {} \;

echo "✅ Build complete!"
echo ""
echo "Test with:"
echo "  python examples/test_validator.py"
