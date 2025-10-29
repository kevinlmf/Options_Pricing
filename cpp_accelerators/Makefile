# Makefile for building C++ accelerated option pricing modules
# Simple alternative to setup.py for development

CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fPIC -Wall
LDFLAGS = -shared

# Python and pybind11 includes
PYTHON_VERSION = $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE = $(shell python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
PYBIND11_INCLUDE = $(shell python3 -m pybind11 --includes)

# OpenMP flags (comment out if not available)
OPENMP_FLAGS = -fopenmp
OPENMP_LIBS = -lgomp

# macOS specific (uncomment if on macOS)
ifeq ($(shell uname -s),Darwin)
	HOMEBREW_PREFIX = $(shell brew --prefix libomp)
	OPENMP_FLAGS = -Xpreprocessor -fopenmp -I$(HOMEBREW_PREFIX)/include
	OPENMP_LIBS = -L$(HOMEBREW_PREFIX)/lib -lomp
	LDFLAGS += -undefined dynamic_lookup
endif

INCLUDES = $(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE) -I.
LIBS = $(OPENMP_LIBS)

# Targets
TARGETS = heston_cpp.so sabr_cpp.so

all: $(TARGETS)

heston_cpp.so: heston_bindings.cpp heston_pricer.hpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(INCLUDES) $(LDFLAGS) -o $@ heston_bindings.cpp $(LIBS)

sabr_cpp.so: sabr_bindings.cpp sabr_pricer.hpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(INCLUDES) $(LDFLAGS) -o $@ sabr_bindings.cpp $(LIBS)

clean:
	rm -f $(TARGETS) *.o

test: all
	python3 test_cpp_modules.py

install: all
	cp $(TARGETS) ../models/

.PHONY: all clean test install
