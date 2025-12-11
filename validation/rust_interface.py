"""
Rust Monte Carlo Validation Engine Interface
============================================

Python interface to the Rust-based Monte Carlo validation engine.
This module bridges Python models with the high-performance Rust backend.
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Path to Rust binary
RUST_BINARY_PATH = os.path.join(
    os.path.dirname(__file__),
    "rust_engine", "target", "release", "volatility_validator"
)

@dataclass
class ValidationResult:
    """Result from Rust validation engine"""
    model_name: str
    rmse: float
    mae: float
    directional_accuracy: float
    sharpe_ratio: float
    cvar_95: float
    net_pnl: float
    arbitrage_violations: int
    martingale_error: float
    execution_time_ms: float
    paths: Optional[np.ndarray] = None


class RustVolatilityValidator:
    """
    Interface to Rust-based volatility validation engine
    """
    
    def __init__(self):
        self._check_rust_binary()
        
    def _check_rust_binary(self):
        """Check if Rust binary exists, if not, try to build it"""
        if not os.path.exists(RUST_BINARY_PATH):
            print("Rust binary not found. Attempting to build...")
            self._build_rust_project()
            
    def _build_rust_project(self):
        """Build the Rust project"""
        rust_path = os.path.join(os.path.dirname(__file__), "rust_engine")
        
        try:
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=rust_path,
                check=True,
                capture_output=True
            )
            print("Rust project built successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to build Rust project: {e.stderr.decode()}")
            raise RuntimeError("Rust build failed")

    def validate_model(self, 
                      model_name: str, 
                      predictions: List[float], 
                      actuals: List[float],
                      market_params: Dict,
                      risk_free_rate: float = 0.02,
                      returns: Optional[List[float]] = None) -> ValidationResult:
        """
        Run validation using Rust engine
        
        Parameters:
        -----------
        model_name : str
            Name of the model being validated
        predictions : list
            Predicted volatility values
        actuals : list
            Actual realized volatility values (or prices/returns for PnL checks)
        market_params : dict
            Market parameters for theoretical checks
        risk_free_rate : float
            Risk free rate for Sharpe Ratio calculation
        returns : list, optional
            Explicit returns series for financial metrics calculation (Sharpe, CVaR, PnL)
            
        Returns:
        --------
        ValidationResult : Validation metrics
        """
        # Prepare input data
        input_data = {
            "model_name": model_name,
            "predictions": predictions,
            "actuals": actuals,
            "market_params": market_params,
            "config": {
                "n_simulations": 10000,
                "check_arbitrage": True
            },
            "risk_free_rate": risk_free_rate,
            "returns": returns
        }
        
        # Serialize to JSON
        json_input = json.dumps(input_data)
        
        # Call Rust binary
        try:
            process = subprocess.run(
                [RUST_BINARY_PATH, "--validate"],
                input=json_input.encode(),
                capture_output=True,
                check=True
            )
            
            # Parse output
            result_json = json.loads(process.stdout.decode())
            
            return ValidationResult(
                model_name=result_json["model_name"],
                rmse=result_json["metrics"]["rmse"],
                mae=result_json["metrics"]["mae"],
                directional_accuracy=result_json["metrics"]["directional_accuracy"],
                sharpe_ratio=result_json["metrics"]["sharpe_ratio"],
                cvar_95=result_json["metrics"]["cvar_95"],
                net_pnl=result_json["metrics"]["net_pnl"],
                arbitrage_violations=result_json["theoretical"]["arbitrage_violations"],
                martingale_error=result_json["theoretical"]["martingale_error"],
                execution_time_ms=result_json["performance"]["execution_time_ms"]
            )
            
        except subprocess.CalledProcessError as e:
            print(f"Rust validation failed: {e.stderr.decode()}")
            raise RuntimeError("Rust validation execution failed")
            
    def run_monte_carlo(self, 
                       initial_price: float, 
                       volatility_model_params: Dict,
                       n_paths: int = 10000,
                       horizon: int = 252) -> np.ndarray:
        """
        Run high-performance Monte Carlo simulation
        """
        input_data = {
            "task": "monte_carlo",
            "initial_price": initial_price,
            "volatility_params": volatility_model_params,
            "n_paths": n_paths,
            "horizon": horizon
        }
        
        json_input = json.dumps(input_data)
        
        try:
            process = subprocess.run(
                [RUST_BINARY_PATH, "--simulate"],
                input=json_input.encode(),
                capture_output=True,
                check=True
            )
            
            result = json.loads(process.stdout.decode())
            return np.array(result["paths"])
            
        except subprocess.CalledProcessError as e:
            print(f"Rust simulation failed: {e.stderr.decode()}")
            raise RuntimeError("Rust simulation execution failed")

