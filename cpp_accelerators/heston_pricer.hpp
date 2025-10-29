#ifndef HESTON_PRICER_HPP
#define HESTON_PRICER_HPP

#include <complex>
#include <cmath>
#include <vector>

/**
 * High-performance Heston model option pricer
 *
 * Implements characteristic function approach with optimized numerical integration
 */
class HestonPricer {
public:
    struct Parameters {
        double S0;      // Spot price
        double K;       // Strike price
        double T;       // Time to maturity
        double r;       // Risk-free rate
        double q;       // Dividend yield
        double v0;      // Initial variance
        double kappa;   // Mean reversion speed
        double theta;   // Long-run variance
        double xi;      // Vol of vol
        double rho;     // Correlation
    };

private:
    Parameters params_;

    // Complex number type
    using Complex = std::complex<double>;

    /**
     * Heston characteristic function
     * @param u Complex frequency parameter
     * @return Characteristic function value
     */
    Complex characteristic_function(const Complex& u) const {
        const double& v0 = params_.v0;
        const double& kappa = params_.kappa;
        const double& theta = params_.theta;
        const double& xi = params_.xi;
        const double& rho = params_.rho;
        const double& r = params_.r;
        const double& q = params_.q;
        const double& T = params_.T;
        const double& S0 = params_.S0;

        // Calculate d
        Complex d = std::sqrt(
            std::pow(rho * xi * u * Complex(0, 1) - kappa, 2.0) +
            xi * xi * (u * Complex(0, 1) + u * u)
        );

        // Calculate g
        Complex g = (kappa - rho * xi * u * Complex(0, 1) - d) /
                    (kappa - rho * xi * u * Complex(0, 1) + d);

        // Calculate C(u, T)
        Complex C = (r - q) * u * Complex(0, 1) * T +
                    (kappa * theta / (xi * xi)) * (
                        (kappa - rho * xi * u * Complex(0, 1) - d) * T -
                        2.0 * std::log((1.0 - g * std::exp(-d * T)) / (1.0 - g))
                    );

        // Calculate D(u, T)
        Complex D = ((kappa - rho * xi * u * Complex(0, 1) - d) / (xi * xi)) *
                    ((1.0 - std::exp(-d * T)) / (1.0 - g * std::exp(-d * T)));

        // Characteristic function
        Complex cf = std::exp(C + D * v0 + Complex(0, 1) * u * std::log(S0));

        return cf;
    }

    /**
     * Integrand for P1 calculation
     */
    double integrand_P1(double phi) const {
        Complex cf = characteristic_function(Complex(phi, -1.0));
        Complex numerator = std::exp(Complex(0, -1) * phi * std::log(params_.K)) * cf;
        Complex denominator = Complex(0, 1) * phi;
        return (numerator / denominator).real();
    }

    /**
     * Integrand for P2 calculation
     */
    double integrand_P2(double phi) const {
        Complex cf = characteristic_function(Complex(phi, 0));
        Complex numerator = std::exp(Complex(0, -1) * phi * std::log(params_.K)) * cf;
        Complex denominator = Complex(0, 1) * phi;
        return (numerator / denominator).real();
    }

    /**
     * Adaptive Simpson's rule for numerical integration
     */
    double adaptive_simpson(double a, double b,
                           double (*f)(const HestonPricer*, double),
                           double epsilon, int max_depth) const {
        double c = (a + b) / 2.0;
        double h = b - a;

        double fa = f(this, a);
        double fb = f(this, b);
        double fc = f(this, c);

        double S = (h / 6.0) * (fa + 4.0 * fc + fb);

        return adaptive_simpson_recursive(a, b, epsilon, S, fa, fb, fc, max_depth, f);
    }

    double adaptive_simpson_recursive(double a, double b, double epsilon,
                                     double S, double fa, double fb, double fc,
                                     int depth,
                                     double (*f)(const HestonPricer*, double)) const {
        if (depth <= 0) return S;

        double c = (a + b) / 2.0;
        double h = b - a;
        double d = (a + c) / 2.0;
        double e = (c + b) / 2.0;

        double fd = f(this, d);
        double fe = f(this, e);

        double Sleft = (h / 12.0) * (fa + 4.0 * fd + fc);
        double Sright = (h / 12.0) * (fc + 4.0 * fe + fb);
        double S2 = Sleft + Sright;

        if (std::abs(S2 - S) <= 15.0 * epsilon) {
            return S2 + (S2 - S) / 15.0;
        }

        return adaptive_simpson_recursive(a, c, epsilon / 2.0, Sleft, fa, fc, fd, depth - 1, f) +
               adaptive_simpson_recursive(c, b, epsilon / 2.0, Sright, fc, fb, fe, depth - 1, f);
    }

    // Static wrapper functions for integration
    static double integrand_P1_wrapper(const HestonPricer* pricer, double phi) {
        return pricer->integrand_P1(phi);
    }

    static double integrand_P2_wrapper(const HestonPricer* pricer, double phi) {
        return pricer->integrand_P2(phi);
    }

public:
    HestonPricer(const Parameters& params) : params_(params) {}

    /**
     * Calculate European call option price
     */
    double call_price() const {
        const double& S0 = params_.S0;
        const double& K = params_.K;
        const double& T = params_.T;
        const double& r = params_.r;
        const double& q = params_.q;

        try {
            // Calculate P1 and P2 via integration
            // Integrate from 0 to 100 (approximates infinity)
            double P1 = 0.5 + (1.0 / M_PI) * adaptive_simpson(
                0.0, 100.0, &integrand_P1_wrapper, 1e-6, 10
            );

            double P2 = 0.5 + (1.0 / M_PI) * adaptive_simpson(
                0.0, 100.0, &integrand_P2_wrapper, 1e-6, 10
            );

            // Call price formula
            double call = S0 * std::exp(-q * T) * P1 - K * std::exp(-r * T) * P2;

            return std::max(call, 0.0);
        } catch (...) {
            // Fallback to Black-Scholes approximation
            return black_scholes_call(S0, K, T, r, q, std::sqrt(params_.v0));
        }
    }

    /**
     * Calculate European put option price
     */
    double put_price() const {
        const double& S0 = params_.S0;
        const double& K = params_.K;
        const double& T = params_.T;
        const double& r = params_.r;
        const double& q = params_.q;

        // Use put-call parity
        double call = call_price();
        double put = call - S0 * std::exp(-q * T) + K * std::exp(-r * T);

        return std::max(put, 0.0);
    }

    /**
     * Batch pricing for multiple options
     * @param options Vector of option parameters
     * @param option_types Vector of option types (true=call, false=put)
     * @return Vector of option prices
     */
    static std::vector<double> batch_price(
        const std::vector<Parameters>& options,
        const std::vector<bool>& is_call
    ) {
        std::vector<double> prices(options.size());

        #pragma omp parallel for
        for (size_t i = 0; i < options.size(); ++i) {
            HestonPricer pricer(options[i]);
            prices[i] = is_call[i] ? pricer.call_price() : pricer.put_price();
        }

        return prices;
    }

private:
    /**
     * Black-Scholes call price (fallback)
     */
    static double black_scholes_call(double S, double K, double T,
                                     double r, double q, double sigma) {
        if (T <= 0) return std::max(S - K, 0.0);

        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
                    (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);

        double call = S * std::exp(-q * T) * norm_cdf(d1) -
                     K * std::exp(-r * T) * norm_cdf(d2);

        return call;
    }

    /**
     * Cumulative normal distribution function
     */
    static double norm_cdf(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }
};

#endif // HESTON_PRICER_HPP
