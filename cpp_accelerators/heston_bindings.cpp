#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "heston_pricer.hpp"

namespace py = pybind11;

/**
 * Python bindings for Heston pricer using pybind11
 */

// Wrapper for single option pricing
py::dict price_heston_option(
    double S0, double K, double T, double r, double q,
    double v0, double kappa, double theta, double xi, double rho,
    const std::string& option_type
) {
    HestonPricer::Parameters params{S0, K, T, r, q, v0, kappa, theta, xi, rho};
    HestonPricer pricer(params);

    double price;
    if (option_type == "call") {
        price = pricer.call_price();
    } else if (option_type == "put") {
        price = pricer.put_price();
    } else {
        throw std::invalid_argument("option_type must be 'call' or 'put'");
    }

    py::dict result;
    result["price"] = price;
    return result;
}

// Wrapper for batch pricing
py::array_t<double> batch_price_heston(
    py::array_t<double> S0_arr,
    py::array_t<double> K_arr,
    py::array_t<double> T_arr,
    py::array_t<double> r_arr,
    py::array_t<double> q_arr,
    py::array_t<double> v0_arr,
    py::array_t<double> kappa_arr,
    py::array_t<double> theta_arr,
    py::array_t<double> xi_arr,
    py::array_t<double> rho_arr,
    py::array_t<bool> is_call_arr
) {
    auto S0 = S0_arr.unchecked<1>();
    auto K = K_arr.unchecked<1>();
    auto T = T_arr.unchecked<1>();
    auto r = r_arr.unchecked<1>();
    auto q = q_arr.unchecked<1>();
    auto v0 = v0_arr.unchecked<1>();
    auto kappa = kappa_arr.unchecked<1>();
    auto theta = theta_arr.unchecked<1>();
    auto xi = xi_arr.unchecked<1>();
    auto rho = rho_arr.unchecked<1>();
    auto is_call = is_call_arr.unchecked<1>();

    size_t n = S0.shape(0);

    // Build parameter vector
    std::vector<HestonPricer::Parameters> options(n);
    std::vector<bool> is_call_vec(n);

    for (size_t i = 0; i < n; ++i) {
        options[i] = HestonPricer::Parameters{
            S0(i), K(i), T(i), r(i), q(i),
            v0(i), kappa(i), theta(i), xi(i), rho(i)
        };
        is_call_vec[i] = is_call(i);
    }

    // Batch price
    std::vector<double> prices = HestonPricer::batch_price(options, is_call_vec);

    // Convert to numpy array
    auto result = py::array_t<double>(n);
    auto result_mut = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; ++i) {
        result_mut(i) = prices[i];
    }

    return result;
}

// Pybind11 module definition
PYBIND11_MODULE(heston_cpp, m) {
    m.doc() = "High-performance Heston model option pricer (C++ accelerated)";

    m.def("price_option", &price_heston_option,
          "Price a single Heston option",
          py::arg("S0"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"),
          py::arg("v0"), py::arg("kappa"), py::arg("theta"),
          py::arg("xi"), py::arg("rho"),
          py::arg("option_type"));

    m.def("batch_price", &batch_price_heston,
          "Price multiple Heston options in parallel",
          py::arg("S0"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("q"),
          py::arg("v0"), py::arg("kappa"), py::arg("theta"),
          py::arg("xi"), py::arg("rho"),
          py::arg("is_call"));
}
