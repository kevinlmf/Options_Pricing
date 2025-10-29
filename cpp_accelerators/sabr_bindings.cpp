#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sabr_pricer.hpp"

namespace py = pybind11;

/**
 * Python bindings for SABR pricer
 */

// Single option pricing
py::dict price_sabr_option(
    double F, double K, double T, double r,
    double alpha, double beta, double rho, double nu,
    const std::string& option_type
) {
    SABRPricer::Parameters params{F, K, T, r, alpha, beta, rho, nu};
    SABRPricer pricer(params);

    double price = pricer.price(option_type == "call");
    double iv = pricer.implied_volatility();

    py::dict result;
    result["price"] = price;
    result["implied_vol"] = iv;
    return result;
}

// Batch pricing
py::array_t<double> batch_price_sabr(
    py::array_t<double> F_arr,
    py::array_t<double> K_arr,
    py::array_t<double> T_arr,
    py::array_t<double> r_arr,
    py::array_t<double> alpha_arr,
    py::array_t<double> beta_arr,
    py::array_t<double> rho_arr,
    py::array_t<double> nu_arr,
    py::array_t<bool> is_call_arr
) {
    auto F = F_arr.unchecked<1>();
    auto K = K_arr.unchecked<1>();
    auto T = T_arr.unchecked<1>();
    auto r = r_arr.unchecked<1>();
    auto alpha = alpha_arr.unchecked<1>();
    auto beta = beta_arr.unchecked<1>();
    auto rho = rho_arr.unchecked<1>();
    auto nu = nu_arr.unchecked<1>();
    auto is_call = is_call_arr.unchecked<1>();

    size_t n = F.shape(0);

    std::vector<SABRPricer::Parameters> options(n);
    std::vector<bool> is_call_vec(n);

    for (size_t i = 0; i < n; ++i) {
        options[i] = SABRPricer::Parameters{
            F(i), K(i), T(i), r(i),
            alpha(i), beta(i), rho(i), nu(i)
        };
        is_call_vec[i] = is_call(i);
    }

    std::vector<double> prices = SABRPricer::batch_price(options, is_call_vec);

    auto result = py::array_t<double>(n);
    auto result_mut = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; ++i) {
        result_mut(i) = prices[i];
    }

    return result;
}

// Batch implied volatility
py::array_t<double> batch_implied_vol_sabr(
    py::array_t<double> F_arr,
    py::array_t<double> K_arr,
    py::array_t<double> T_arr,
    py::array_t<double> r_arr,
    py::array_t<double> alpha_arr,
    py::array_t<double> beta_arr,
    py::array_t<double> rho_arr,
    py::array_t<double> nu_arr
) {
    auto F = F_arr.unchecked<1>();
    auto K = K_arr.unchecked<1>();
    auto T = T_arr.unchecked<1>();
    auto r = r_arr.unchecked<1>();
    auto alpha = alpha_arr.unchecked<1>();
    auto beta = beta_arr.unchecked<1>();
    auto rho = rho_arr.unchecked<1>();
    auto nu = nu_arr.unchecked<1>();

    size_t n = F.shape(0);

    std::vector<SABRPricer::Parameters> options(n);

    for (size_t i = 0; i < n; ++i) {
        options[i] = SABRPricer::Parameters{
            F(i), K(i), T(i), r(i),
            alpha(i), beta(i), rho(i), nu(i)
        };
    }

    std::vector<double> ivs = SABRPricer::batch_implied_vol(options);

    auto result = py::array_t<double>(n);
    auto result_mut = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; ++i) {
        result_mut(i) = ivs[i];
    }

    return result;
}

PYBIND11_MODULE(sabr_cpp, m) {
    m.doc() = "High-performance SABR model option pricer (C++ accelerated)";

    m.def("price_option", &price_sabr_option,
          "Price a single SABR option",
          py::arg("F"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"),
          py::arg("option_type"));

    m.def("batch_price", &batch_price_sabr,
          "Price multiple SABR options in parallel",
          py::arg("F"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"),
          py::arg("is_call"));

    m.def("batch_implied_vol", &batch_implied_vol_sabr,
          "Calculate implied volatility for multiple SABR options in parallel",
          py::arg("F"), py::arg("K"), py::arg("T"), py::arg("r"),
          py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"));
}
