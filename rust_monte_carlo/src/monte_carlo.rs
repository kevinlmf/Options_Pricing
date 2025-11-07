use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use crate::models::*;
use crate::statistics::*;

/// High-performance Monte Carlo simulator
pub struct MonteCarloEngine {
    params: SimulationParams,
    seed: Option<u64>,
}

impl MonteCarloEngine {
    /// Create new Monte Carlo engine
    pub fn new(params: SimulationParams) -> Self {
        Self { params, seed: None }
    }

    /// Create with fixed seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run parallel Monte Carlo simulations
    pub fn run_simulations(&self) -> Vec<Vec<f64>> {
        (0..self.params.n_simulations)
            .into_par_iter()
            .map(|i| {
                let seed = self.seed.unwrap_or(i as u64);
                self.simulate_single_path(seed)
            })
            .collect()
    }

    /// Simulate a single price path using Geometric Brownian Motion
    fn simulate_single_path(&self, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut path = Vec::with_capacity(self.params.n_steps + 1);
        path.push(self.params.initial_price);

        let dt = self.params.dt;
        let drift = self.params.drift;
        let vol = self.params.volatility;

        let mut price = self.params.initial_price;

        for _ in 0..self.params.n_steps {
            let z = normal.sample(&mut rng);

            // Geometric Brownian Motion: dS = μ*S*dt + σ*S*dW
            let drift_term = drift * dt;
            let diffusion_term = vol * dt.sqrt() * z;

            price *= (drift_term + diffusion_term).exp();
            path.push(price);
        }

        path
    }

    /// Validate agent prediction against simulated paths
    pub fn validate_prediction(
        &self,
        prediction: &AgentPrediction,
        confidence_level: f64,
    ) -> ValidationResult {
        // Run simulations
        let paths = self.run_simulations();

        // Extract final prices
        let final_prices: Vec<f64> = paths
            .iter()
            .map(|path| *path.last().unwrap())
            .collect();

        // Calculate statistics
        let stats = calculate_statistics(&final_prices);

        // Calculate confidence interval
        let ci = calculate_confidence_interval(&final_prices, confidence_level);

        // Calculate errors
        let mean_error = (stats.mean - prediction.predicted_mean).abs();
        let std_error = (stats.std - prediction.predicted_std).abs();

        // Statistical test (simplified z-test)
        let z_score = (prediction.predicted_mean - stats.mean) / (stats.std / (final_prices.len() as f64).sqrt());
        let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));

        // Validation decision
        let is_valid = p_value > 0.05 &&
                       prediction.predicted_mean >= ci.0 &&
                       prediction.predicted_mean <= ci.1;

        ValidationResult {
            agent_id: prediction.agent_id.clone(),
            mean_error,
            std_error,
            confidence_interval: ci,
            is_valid,
            p_value,
            simulated_paths: paths,
            statistics: ValidationStatistics {
                mean: stats.mean,
                median: stats.median,
                std: stats.std,
                skewness: stats.skewness,
                kurtosis: stats.kurtosis,
                var_95: stats.var_95,
                cvar_95: stats.cvar_95,
                min: stats.min,
                max: stats.max,
            },
        }
    }

    /// Validate multiple agents in parallel
    pub fn validate_multiple_agents(
        &self,
        predictions: &[AgentPrediction],
        confidence_level: f64,
    ) -> Vec<ValidationResult> {
        predictions
            .par_iter()
            .map(|pred| self.validate_prediction(pred, confidence_level))
            .collect()
    }

    /// Run scenario analysis with different volatility regimes
    pub fn scenario_analysis(
        &self,
        volatility_scenarios: &[f64],
    ) -> Vec<(f64, ValidationStatistics)> {
        volatility_scenarios
            .par_iter()
            .map(|&vol| {
                let mut params = self.params.clone();
                params.volatility = vol;
                let engine = MonteCarloEngine::new(params);
                let paths = engine.run_simulations();
                let final_prices: Vec<f64> = paths
                    .iter()
                    .map(|path| *path.last().unwrap())
                    .collect();
                let stats = calculate_statistics(&final_prices);
                (vol, ValidationStatistics {
                    mean: stats.mean,
                    median: stats.median,
                    std: stats.std,
                    skewness: stats.skewness,
                    kurtosis: stats.kurtosis,
                    var_95: stats.var_95,
                    cvar_95: stats.cvar_95,
                    min: stats.min,
                    max: stats.max,
                })
            })
            .collect()
    }

    /// Calculate Greeks using finite differences
    pub fn calculate_greeks(&self, option_type: &str, strike: f64) -> Greeks {
        let paths = self.run_simulations();
        let final_prices: Vec<f64> = paths
            .iter()
            .map(|path| *path.last().unwrap())
            .collect();

        let payoffs: Vec<f64> = final_prices
            .iter()
            .map(|&price| match option_type {
                "call" => (price - strike).max(0.0),
                "put" => (strike - price).max(0.0),
                _ => 0.0,
            })
            .collect();

        let option_value = payoffs.iter().sum::<f64>() / payoffs.len() as f64;

        // Delta calculation (simplified)
        let delta_bump = 0.01 * self.params.initial_price;
        let mut params_up = self.params.clone();
        params_up.initial_price += delta_bump;
        let engine_up = MonteCarloEngine::new(params_up);
        let paths_up = engine_up.run_simulations();
        let final_prices_up: Vec<f64> = paths_up
            .iter()
            .map(|path| *path.last().unwrap())
            .collect();
        let payoffs_up: Vec<f64> = final_prices_up
            .iter()
            .map(|&price| match option_type {
                "call" => (price - strike).max(0.0),
                "put" => (strike - price).max(0.0),
                _ => 0.0,
            })
            .collect();
        let value_up = payoffs_up.iter().sum::<f64>() / payoffs_up.len() as f64;
        let delta = (value_up - option_value) / delta_bump;

        Greeks {
            delta,
            gamma: 0.0,  // Would need second-order finite difference
            vega: 0.0,   // Would need volatility bump
            theta: 0.0,  // Would need time bump
        }
    }
}

/// Option Greeks
#[derive(Debug, Clone)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
}

/// Normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_simulation() {
        let params = SimulationParams::default();
        let engine = MonteCarloEngine::new(params).with_seed(42);
        let path = engine.simulate_single_path(42);

        assert_eq!(path.len(), 101);
        assert_eq!(path[0], 100.0);
        assert!(path.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_parallel_simulations() {
        let params = SimulationParams {
            n_simulations: 1000,
            n_steps: 50,
            ..Default::default()
        };
        let engine = MonteCarloEngine::new(params);
        let paths = engine.run_simulations();

        assert_eq!(paths.len(), 1000);
        assert!(paths.iter().all(|p| p.len() == 51));
    }
}
