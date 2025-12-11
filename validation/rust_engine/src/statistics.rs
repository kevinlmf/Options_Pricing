use statrs::statistics::{OrderStatistics, Statistics};

/// Comprehensive statistics for a data series
#[derive(Debug, Clone)]
pub struct DetailedStatistics {
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

/// Calculate comprehensive statistics
pub fn calculate_statistics(data: &[f64]) -> DetailedStatistics {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = data.mean();
    // Calculate median manually
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let std = data.std_dev();

    // Skewness and Kurtosis
    let skewness = calculate_skewness(data, mean, std);
    let kurtosis = calculate_kurtosis(data, mean, std);

    // VaR and CVaR at 95% confidence
    let var_95 = calculate_var(&sorted, 0.05);
    let cvar_95 = calculate_cvar(&sorted, 0.05);

    DetailedStatistics {
        mean,
        median,
        std,
        skewness,
        kurtosis,
        var_95,
        cvar_95,
        min: *sorted.first().unwrap(),
        max: *sorted.last().unwrap(),
    }
}

/// Calculate skewness
fn calculate_skewness(data: &[f64], mean: f64, std: f64) -> f64 {
    let n = data.len() as f64;
    let sum_cubed_deviations: f64 = data
        .iter()
        .map(|&x| ((x - mean) / std).powi(3))
        .sum();

    sum_cubed_deviations / n
}

/// Calculate kurtosis
fn calculate_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
    let n = data.len() as f64;
    let sum_fourth_deviations: f64 = data
        .iter()
        .map(|&x| ((x - mean) / std).powi(4))
        .sum();

    (sum_fourth_deviations / n) - 3.0  // Excess kurtosis
}

/// Calculate Value at Risk (VaR)
pub fn calculate_var(sorted_data: &[f64], alpha: f64) -> f64 {
    let index = (alpha * sorted_data.len() as f64).floor() as usize;
    sorted_data[index]
}

/// Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
pub fn calculate_cvar(sorted_data: &[f64], alpha: f64) -> f64 {
    let index = (alpha * sorted_data.len() as f64).floor() as usize;
    let tail = &sorted_data[..=index];
    tail.iter().sum::<f64>() / tail.len() as f64
}

/// Calculate confidence interval
pub fn calculate_confidence_interval(data: &[f64], confidence: f64) -> (f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = (1.0 - confidence) / 2.0;
    let lower_idx = (alpha * sorted.len() as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha) * sorted.len() as f64).ceil() as usize - 1;

    (sorted[lower_idx], sorted[upper_idx])
}

/// Calculate returns from price series
pub fn calculate_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Calculate realized volatility
pub fn calculate_realized_volatility(returns: &[f64], annualization_factor: f64) -> f64 {
    let variance = returns.variance();
    (variance * annualization_factor).sqrt()
}

/// Calculate Sharpe ratio
pub fn calculate_sharpe_ratio(returns: &[f64], risk_free_rate: f64, annualization_factor: f64) -> f64 {
    let mean_return = returns.mean();
    let std_return = returns.std_dev();

    let annualized_return = mean_return * annualization_factor;
    let annualized_vol = std_return * annualization_factor.sqrt();

    (annualized_return - risk_free_rate) / annualized_vol
}

/// Calculate maximum drawdown
pub fn calculate_max_drawdown(prices: &[f64]) -> f64 {
    let mut max_price = prices[0];
    let mut max_drawdown = 0.0;

    for &price in prices.iter() {
        if price > max_price {
            max_price = price;
        }
        let drawdown = (max_price - price) / max_price;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    max_drawdown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_statistics(&data);

        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_var_cvar() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let var = calculate_var(&data, 0.05);
        let cvar = calculate_cvar(&data, 0.05);

        assert!(var >= 1.0 && var <= 2.0);
        assert!(cvar >= 1.0 && cvar <= 1.5);
    }
}
