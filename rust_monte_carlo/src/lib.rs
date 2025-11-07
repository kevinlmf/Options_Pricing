mod models;
mod monte_carlo;
mod statistics;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use models::*;
use monte_carlo::*;
use statistics::*;

/// Python wrapper for MonteCarloEngine
#[pyclass]
struct PyMonteCarloEngine {
    engine: MonteCarloEngine,
}

#[pymethods]
impl PyMonteCarloEngine {
    #[new]
    fn new(
        n_simulations: usize,
        n_steps: usize,
        dt: f64,
        initial_price: f64,
        drift: f64,
        volatility: f64,
    ) -> Self {
        let params = SimulationParams {
            n_simulations,
            n_steps,
            dt,
            initial_price,
            drift,
            volatility,
        };
        Self {
            engine: MonteCarloEngine::new(params),
        }
    }

    /// Run Monte Carlo simulations
    fn run_simulations(&self, py: Python) -> PyResult<PyObject> {
        let paths = self.engine.run_simulations();

        // Convert to Python list of lists
        let py_paths = PyList::empty(py);
        for path in paths.iter() {
            let py_path = PyList::new(py, path);
            py_paths.append(py_path)?;
        }

        Ok(py_paths.into())
    }

    /// Validate agent prediction
    fn validate_prediction(
        &self,
        py: Python,
        agent_id: String,
        predicted_mean: f64,
        predicted_std: f64,
        confidence: f64,
        confidence_level: f64,
    ) -> PyResult<PyObject> {
        let prediction = AgentPrediction {
            agent_id,
            predicted_mean,
            predicted_std,
            confidence,
            regime: MarketRegime::LowVol,  // Default
        };

        let result = self.engine.validate_prediction(&prediction, confidence_level);

        // Convert to Python dict
        let dict = PyDict::new(py);
        dict.set_item("agent_id", result.agent_id)?;
        dict.set_item("mean_error", result.mean_error)?;
        dict.set_item("std_error", result.std_error)?;
        dict.set_item("confidence_interval", result.confidence_interval)?;
        dict.set_item("is_valid", result.is_valid)?;
        dict.set_item("p_value", result.p_value)?;

        // Statistics
        let stats = PyDict::new(py);
        stats.set_item("mean", result.statistics.mean)?;
        stats.set_item("median", result.statistics.median)?;
        stats.set_item("std", result.statistics.std)?;
        stats.set_item("skewness", result.statistics.skewness)?;
        stats.set_item("kurtosis", result.statistics.kurtosis)?;
        stats.set_item("var_95", result.statistics.var_95)?;
        stats.set_item("cvar_95", result.statistics.cvar_95)?;
        stats.set_item("min", result.statistics.min)?;
        stats.set_item("max", result.statistics.max)?;

        dict.set_item("statistics", stats)?;

        Ok(dict.into())
    }

    /// Run scenario analysis
    fn scenario_analysis(&self, py: Python, volatilities: Vec<f64>) -> PyResult<PyObject> {
        let results = self.engine.scenario_analysis(&volatilities);

        let py_results = PyList::empty(py);
        for (vol, stats) in results {
            let item = PyDict::new(py);
            item.set_item("volatility", vol)?;
            item.set_item("mean", stats.mean)?;
            item.set_item("std", stats.std)?;
            item.set_item("var_95", stats.var_95)?;
            item.set_item("cvar_95", stats.cvar_95)?;
            py_results.append(item)?;
        }

        Ok(py_results.into())
    }

    /// Calculate option Greeks
    fn calculate_greeks(
        &self,
        py: Python,
        option_type: String,
        strike: f64,
    ) -> PyResult<PyObject> {
        let greeks = self.engine.calculate_greeks(&option_type, strike);

        let dict = PyDict::new(py);
        dict.set_item("delta", greeks.delta)?;
        dict.set_item("gamma", greeks.gamma)?;
        dict.set_item("vega", greeks.vega)?;
        dict.set_item("theta", greeks.theta)?;

        Ok(dict.into())
    }
}

/// Python module initialization
#[pymodule]
fn monte_carlo_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMonteCarloEngine>()?;

    // Add module-level functions
    #[pyfn(m)]
    fn quick_validate(
        _py: Python,
        n_simulations: usize,
        predicted_mean: f64,
        predicted_std: f64,
        initial_price: f64,
        drift: f64,
        volatility: f64,
    ) -> PyResult<bool> {
        let params = SimulationParams {
            n_simulations,
            n_steps: 100,
            dt: 1.0 / 252.0,
            initial_price,
            drift,
            volatility,
        };

        let engine = MonteCarloEngine::new(params);
        let prediction = AgentPrediction {
            agent_id: "test".to_string(),
            predicted_mean,
            predicted_std,
            confidence: 0.95,
            regime: MarketRegime::LowVol,
        };

        let result = engine.validate_prediction(&prediction, 0.95);
        Ok(result.is_valid)
    }

    #[pyfn(m)]
    fn batch_validate(
        _py: Python,
        n_simulations: usize,
        predictions: Vec<(f64, f64)>,  // (mean, std) pairs
        initial_price: f64,
        drift: f64,
        volatility: f64,
    ) -> PyResult<Vec<bool>> {
        let params = SimulationParams {
            n_simulations,
            n_steps: 100,
            dt: 1.0 / 252.0,
            initial_price,
            drift,
            volatility,
        };

        let engine = MonteCarloEngine::new(params);

        let agent_predictions: Vec<AgentPrediction> = predictions
            .iter()
            .enumerate()
            .map(|(i, &(mean, std))| AgentPrediction {
                agent_id: format!("agent_{}", i),
                predicted_mean: mean,
                predicted_std: std,
                confidence: 0.95,
                regime: MarketRegime::LowVol,
            })
            .collect();

        let results = engine.validate_multiple_agents(&agent_predictions, 0.95);
        Ok(results.iter().map(|r| r.is_valid).collect())
    }

    Ok(())
}
