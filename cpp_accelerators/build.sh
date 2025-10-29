#!/bin/bash
# Build script for C++ accelerated option pricing modules

set -e

echo "================================================"
echo "Building C++ Accelerated Option Pricing Modules"
echo "================================================"
echo

# Check for pybind11
echo "Checking dependencies..."
python3 -c "import pybind11" 2>/dev/null || {
    echo "Error: pybind11 not found"
    echo "Install with: pip install pybind11"
    exit 1
}
echo "✓ pybind11 found"

# Check for compiler
if command -v g++ &> /dev/null; then
    echo "✓ g++ found: $(g++ --version | head -1)"
elif command -v clang++ &> /dev/null; then
    echo "✓ clang++ found: $(clang++ --version | head -1)"
else
    echo "Error: No C++ compiler found"
    exit 1
fi

# Build method selection
echo
echo "Select build method:"
echo "1) Make (fastest, recommended for development)"
echo "2) setup.py (recommended for installation)"
echo "3) CMake (advanced)"
read -p "Choice [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo
        echo "Building with Make..."
        make clean
        make
        echo "✓ Build complete"
        ;;
    2)
        echo
        echo "Building with setup.py..."
        python3 setup.py build_ext --inplace
        echo "✓ Build complete"
        ;;
    3)
        echo
        echo "Building with CMake..."
        mkdir -p build
        cd build
        cmake ..
        make
        cp *.so ..
        cd ..
        echo "✓ Build complete"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Run tests
echo
read -p "Run tests? [Y/n]: " run_tests
run_tests=${run_tests:-Y}

if [[ $run_tests =~ ^[Yy] ]]; then
    echo
    echo "Running tests..."
    python3 test_cpp_modules.py
fi

echo
echo "================================================"
echo "Build successful!"
echo "================================================"
echo
echo "To use the modules in your Python code:"
echo "  import heston_cpp"
echo "  import sabr_cpp"
echo
echo "See README.md for usage examples"
