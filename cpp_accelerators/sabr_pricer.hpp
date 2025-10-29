#ifndef SABR_PRICER_HPP
#define SABR_PRICER_HPP

#include <cmath>
#include <vector>
#include <algorithm>

/**
 * High-performance SABR model option pricer
 *
 * Implements Hagan's approximation formula for implied volatility
 */
class SABRPricer {
public:
    struct Parameters {
        double F;       // Forward price
        double K;       // Strike price
        double T;       // Time to maturity
        double r;       // Risk-free rate
        double alpha;   // Initial volatility
        double beta;    // CEV parameter
        double rho;     // Correlation
        double nu;      // Vol of vol
    };

private:
    Parameters params_;

    /**
     * ATM implied volatility
     */
    double atm_implied_volatility() const {
        const double& F = params_.F;
        const double& alpha = params_.alpha;
        const double& beta = params_.beta;
        const double& rho = params_.rho;
        const double& nu = params_.nu;
        const double& T = params_.T;

        double F_power = std::pow(F, 1.0 - beta);
        double term1 = alpha / F_power;

        double term2 = 1.0 + T * (
            ((1.0 - beta) * (1.0 - beta) / 24.0) * (alpha * alpha / (F_power * F_power)) +
            (rho * beta * nu * alpha / (4.0 * F_power)) +
            ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu
        );

        return term1 * term2;
    }

public:
    SABRPricer(const Parameters& params) : params_(params) {}

    /**
     * Calculate implied volatility using Hagan's approximation
     */
    double implied_volatility() const {
        const double& F = params_.F;
        const double& K = params_.K;
        const double& T = params_.T;
        const double& alpha = params_.alpha;
        const double& beta = params_.beta;
        const double& rho = params_.rho;
        const double& nu = params_.nu;

        // Handle ATM case
        if (std::abs(F - K) < 1e-10) {
            return atm_implied_volatility();
        }

        // Calculate intermediate values
        double FK = F * K;
        double log_FK = std::log(F / K);

        // z parameter
        double FK_power = std::pow(FK, (1.0 - beta) / 2.0);
        double z = (nu / alpha) * FK_power * log_FK;

        // x(z) function
        double x_z;
        if (std::abs(z) < 1e-6) {
            x_z = 1.0;
        } else {
            double sqrt_term = std::sqrt(1.0 - 2.0 * rho * z + z * z);
            double numerator = std::log((sqrt_term + z - rho) / (1.0 - rho));
            x_z = z / numerator;
        }

        // Term 1: Main term
        double log_FK_sq = log_FK * log_FK;
        double log_FK_4 = log_FK_sq * log_FK_sq;

        double term1_denom = FK_power * (
            1.0 + ((1.0 - beta) * (1.0 - beta) / 24.0) * log_FK_sq +
            ((1.0 - beta) * (1.0 - beta) * (1.0 - beta) * (1.0 - beta) / 1920.0) * log_FK_4
        );
        double term1 = alpha / term1_denom;

        // Term 2: Time-dependent correction
        double FK_power_full = std::pow(FK, 1.0 - beta);
        double term2 = 1.0 + T * (
            ((1.0 - beta) * (1.0 - beta) / 24.0) * (alpha * alpha / FK_power_full) +
            (rho * beta * nu * alpha / (4.0 * FK_power)) +
            ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu
        );

        double implied_vol = term1 * x_z * term2;

        return std::max(implied_vol, 1e-6);
    }

    /**
     * Calculate option price using SABR implied volatility
     */
    double price(bool is_call = true) const {
        double iv = implied_volatility();

        // Convert forward to spot
        double S = params_.F * std::exp(-params_.r * params_.T);

        return black_scholes_price(S, params_.K, params_.T, params_.r, iv, is_call);
    }

    /**
     * Batch pricing for multiple options
     */
    static std::vector<double> batch_price(
        const std::vector<Parameters>& options,
        const std::vector<bool>& is_call
    ) {
        std::vector<double> prices(options.size());

        #pragma omp parallel for
        for (size_t i = 0; i < options.size(); ++i) {
            SABRPricer pricer(options[i]);
            prices[i] = pricer.price(is_call[i]);
        }

        return prices;
    }

    /**
     * Batch implied volatility calculation
     */
    static std::vector<double> batch_implied_vol(
        const std::vector<Parameters>& options
    ) {
        std::vector<double> ivs(options.size());

        #pragma omp parallel for
        for (size_t i = 0; i < options.size(); ++i) {
            SABRPricer pricer(options[i]);
            ivs[i] = pricer.implied_volatility();
        }

        return ivs;
    }

private:
    /**
     * Black-Scholes pricing formula
     */
    static double black_scholes_price(double S, double K, double T,
                                      double r, double sigma, bool is_call) {
        if (T <= 0) return std::max(is_call ? (S - K) : (K - S), 0.0);

        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
                    (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);

        if (is_call) {
            return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
        } else {
            return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
        }
    }

    /**
     * Cumulative normal distribution function
     */
    static double norm_cdf(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }
};

#endif // SABR_PRICER_HPP
