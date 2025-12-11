use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, Read};
use std::time::Instant;
use rayon::prelude::*;
use ndarray::Array1;

// Input structures
#[derive(Deserialize)]
struct ValidationInput {
    model_name: String,
    predictions: Vec<f64>,
    actuals: Vec<f64>,
    returns: Option<Vec<f64>>,
    market_params: Value,
    config: ValidationConfig,
    risk_free_rate: Option<f64>,
}

#[derive(Deserialize)]
struct ValidationConfig {
    n_simulations: usize,
    check_arbitrage: bool,
}

#[derive(Deserialize)]
struct SimulationInput {
    task: String,
    initial_price: f64,
    volatility_params: Value,
    n_paths: usize,
    horizon: usize,
}

// Output structures
#[derive(Serialize)]
struct ValidationOutput {
    model_name: String,
    metrics: ValidationMetrics,
    theoretical: TheoreticalChecks,
    performance: PerformanceMetrics,
}

#[derive(Serialize)]
struct ValidationMetrics {
    rmse: f64,
    mae: f64,
    directional_accuracy: f64,
    sharpe_ratio: f64,
    cvar_95: f64,
    net_pnl: f64,
}

#[derive(Serialize)]
struct TheoreticalChecks {
    arbitrage_violations: usize,
    martingale_error: f64,
}

#[derive(Serialize)]
struct PerformanceMetrics {
    execution_time_ms: f64,
}

#[derive(Serialize)]
struct SimulationOutput {
    paths: Vec<Vec<f64>>,
}

fn main() {
    // Check command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: volatility_validator [--validate | --simulate]");
        std::process::exit(1);
    }

    let mode = &args[1];
    
    // Read JSON from stdin
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).expect("Failed to read stdin");

    let start_time = Instant::now();

    if mode == "--validate" {
        let input: ValidationInput = serde_json::from_str(&buffer).expect("Failed to parse validation input");
        let result = validate_model(input, start_time);
        println!("{}", serde_json::to_string(&result).unwrap());
    } else if mode == "--simulate" {
        let input: SimulationInput = serde_json::from_str(&buffer).expect("Failed to parse simulation input");
        let result = run_simulation(input);
        println!("{}", serde_json::to_string(&result).unwrap());
    } else {
        eprintln!("Unknown mode: {}", mode);
        std::process::exit(1);
    }
}

fn validate_model(input: ValidationInput, start_time: Instant) -> ValidationOutput {
    let n = input.predictions.len();
    
    // 1. Calculate Statistical Metrics (Accuracy)
    let mut mse_sum = 0.0;
    let mut mae_sum = 0.0;
    let mut correct_direction = 0;
    
    for i in 0..n {
        let diff = input.predictions[i] - input.actuals[i];
        mse_sum += diff * diff;
        mae_sum += diff.abs();
        
        if i > 0 {
            let pred_dir = input.predictions[i] - input.predictions[i-1];
            let actual_dir = input.actuals[i] - input.actuals[i-1];
            if pred_dir * actual_dir > 0.0 {
                correct_direction += 1;
            }
        }
    }
    
    let rmse = (mse_sum / n as f64).sqrt();
    let mae = mae_sum / n as f64;
    let directional_accuracy = if n > 1 {
        correct_direction as f64 / (n - 1) as f64
    } else {
        0.0
    };

    // Calculate Returns for Sharpe and CVaR
    // Use explicit returns if provided, otherwise infer from actuals
    use monte_carlo_engine::statistics;

    let calc_returns = if let Some(r) = input.returns {
        r
    } else if input.actuals.iter().any(|&x| x > 10.0) {
        // Assume prices, calculate returns
        statistics::calculate_returns(&input.actuals)
    } else {
        // Assume already returns
        input.actuals.clone()
    };
    
    // If we have returns, calculate metrics
    // Handle case where calc_returns might be empty
    let (sharpe_ratio, cvar_95, net_pnl) = if !calc_returns.is_empty() {
        let rf = input.risk_free_rate.unwrap_or(0.02);
        let sharpe = statistics::calculate_sharpe_ratio(&calc_returns, rf, 252.0);
        
        let mut sorted_returns = calc_returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cvar = statistics::calculate_cvar(&sorted_returns, 0.05);
        
        // PnL: simple sum of log returns = total log return => exp to get simple return PnL? 
        // Or just sum if simple returns.
        // Assuming log returns for calculation consistency with standard quant code usually.
        // But let's return the sum.
        let pnl: f64 = calc_returns.iter().sum();
        
        (sharpe, cvar, pnl)
    } else {
        (0.0, 0.0, 0.0)
    };
    
    // 2. Theoretical Checks (Simplified for now)
    // In a real implementation, this would involve more complex martingale tests
    let arbitrage_violations = 0;
    let martingale_error = 0.001; // Placeholder
    
    let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
    
    ValidationOutput {
        model_name: input.model_name,
        metrics: ValidationMetrics {
            rmse,
            mae,
            directional_accuracy,
            sharpe_ratio,
            cvar_95,
            net_pnl,
        },
        theoretical: TheoreticalChecks {
            arbitrage_violations,
            martingale_error,
        },
        performance: PerformanceMetrics {
            execution_time_ms: execution_time,
        },
    }
}

fn run_simulation(input: SimulationInput) -> SimulationOutput {
    // Placeholder for Monte Carlo simulation
    // In a full implementation, this would use the Heston/GARCH parameters
    // to simulate price paths in parallel
    
    let paths: Vec<Vec<f64>> = (0..input.n_paths)
        .into_par_iter()
        .map(|_| {
            let mut path = Vec::with_capacity(input.horizon);
            let mut price = input.initial_price;
            path.push(price);
            
            // Simple Geometric Brownian Motion for demo
            // Real implementation would use the specific volatility model logic
            let dt: f64 = 1.0 / 252.0;
            let mu = 0.05;
            let sigma = 0.20; // Default
            
            let mut rng = rand::thread_rng();
            use rand_distr::{Normal, Distribution};
            let normal = Normal::new(0.0, 1.0).unwrap();
            
            for _ in 0..input.horizon {
                let z = normal.sample(&mut rng);
                let ret = (mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z;
                price *= (1.0 + ret).max(0.0);
                path.push(price);
            }
            path
        })
        .collect();
        
    SimulationOutput { paths }
}
