use serde::{Deserialize, Serialize};

/// Market regime types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MarketRegime {
    LowVol,
    HighVol,
    Trending,
    MeanReverting,
    Crisis,
}

/// Agent prediction structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPrediction {
    pub agent_id: String,
    pub predicted_mean: f64,
    pub predicted_std: f64,
    pub confidence: f64,
    pub regime: MarketRegime,
}

/// Monte Carlo simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    pub n_simulations: usize,
    pub n_steps: usize,
    pub dt: f64,
    pub initial_price: f64,
    pub drift: f64,
    pub volatility: f64,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub agent_id: String,
    pub mean_error: f64,
    pub std_error: f64,
    pub confidence_interval: (f64, f64),
    pub is_valid: bool,
    pub p_value: f64,
    pub simulated_paths: Vec<Vec<f64>>,
    pub statistics: ValidationStatistics,
}

/// Detailed validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub min: f64,
    pub max: f64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            n_simulations: 10_000,
            n_steps: 100,
            dt: 1.0 / 252.0,  // Daily
            initial_price: 100.0,
            drift: 0.0,
            volatility: 0.2,
        }
    }
}
