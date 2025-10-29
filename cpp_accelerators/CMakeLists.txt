cmake_minimum_required(VERSION 3.12)
project(option_pricing_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find OpenMP
find_package(OpenMP)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Heston module
pybind11_add_module(heston_cpp heston_bindings.cpp)
target_include_directories(heston_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(heston_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# SABR module
pybind11_add_module(sabr_cpp sabr_bindings.cpp)
target_include_directories(sabr_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(sabr_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install targets
install(TARGETS heston_cpp sabr_cpp
        LIBRARY DESTINATION ${Python3_SITELIB})
