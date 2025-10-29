"""
Visualization and Reporting Module

Generate comprehensive visualizations and reports for model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelComparisonVisualizer:
    """
    Create visualizations for model comparison results
    """

    def __init__(self, results: Dict, save_dir: str = "./results"):
        """
        Initialize visualizer

        Parameters:
        -----------
        results : dict
            Dictionary of BacktestResult objects by model name
        save_dir : str
            Directory to save visualizations
        """
        self.results = results
        self.save_dir = save_dir

        import os
        os.makedirs(save_dir, exist_ok=True)

    def plot_pnl_comparison(self, figsize: tuple = (14, 8)):
        """
        Plot P&L comparison across all models

        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance Comparison - P&L Analysis', fontsize=16, fontweight='bold')

        # 1. Cumulative P&L over time
        ax1 = axes[0, 0]
        for model_name, result in self.results.items():
            if not result.pnl_series.empty:
                cumulative_pnl = result.pnl_series.cumsum()
                ax1.plot(cumulative_pnl.index, cumulative_pnl.values,
                        label=model_name.upper(), linewidth=2, alpha=0.8)

        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Cumulative P&L ($)')
        ax1.set_title('Cumulative P&L Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # 2. Final P&L bar chart
        ax2 = axes[0, 1]
        models = []
        pnls = []
        colors = []

        for model_name, result in self.results.items():
            models.append(model_name.upper())
            pnls.append(result.total_pnl)
            colors.append('green' if result.total_pnl > 0 else 'red')

        bars = ax2.bar(models, pnls, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Total P&L ($)')
        ax2.set_title('Final Total P&L by Model')
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')

        # 3. Sharpe Ratio comparison
        ax3 = axes[1, 0]
        models = []
        sharpes = []

        for model_name, result in self.results.items():
            models.append(model_name.upper())
            sharpes.append(result.sharpe_ratio)

        bars = ax3.bar(models, sharpes, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Sharpe Ratio Comparison')
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Threshold (1.0)')
        ax3.legend()

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')

        # 4. Win Rate and Trade Count
        ax4 = axes[1, 1]
        models = []
        win_rates = []
        trade_counts = []

        for model_name, result in self.results.items():
            models.append(model_name.upper())
            win_rates.append(result.win_rate)
            trade_counts.append(result.num_trades)

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax4.bar(x - width/2, win_rates, width, label='Win Rate (%)',
                       color='lightgreen', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x + width/2, trade_counts, width, label='Num Trades',
                       color='lightcoral', alpha=0.7, edgecolor='black')

        ax4.set_ylabel('Value')
        ax4.set_title('Win Rate & Trade Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/pnl_comparison.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved P&L comparison plot to {self.save_dir}/pnl_comparison.png")
        plt.close()

    def plot_risk_metrics(self, figsize: tuple = (14, 6)):
        """
        Plot risk metrics comparison

        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Risk Metrics Comparison', fontsize=16, fontweight='bold')

        # 1. Maximum Drawdown
        ax1 = axes[0]
        models = []
        drawdowns = []

        for model_name, result in self.results.items():
            models.append(model_name.upper())
            drawdowns.append(abs(result.max_drawdown))

        bars = ax1.bar(models, drawdowns, color='indianred', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Maximum Drawdown ($)')
        ax1.set_title('Maximum Drawdown (Lower is Better)')
        ax1.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom', fontsize=9)

        # 2. VaR 95%
        ax2 = axes[1]
        models = []
        vars = []

        for model_name, result in self.results.items():
            models.append(model_name.upper())
            vars.append(abs(result.var_95))

        bars = ax2.bar(models, vars, color='darkorange', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('VaR 95% ($)')
        ax2.set_title('Value at Risk (95% confidence)')
        ax2.grid(True, axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom', fontsize=9)

        # 3. Transaction Costs
        ax3 = axes[2]
        models = []
        trans_costs = []
        hedge_costs = []

        for model_name, result in self.results.items():
            models.append(model_name.upper())
            trans_costs.append(result.transaction_costs)
            hedge_costs.append(result.hedge_costs)

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax3.bar(x - width/2, trans_costs, width, label='Transaction Costs',
                       color='lightblue', alpha=0.7, edgecolor='black')
        bars2 = ax3.bar(x + width/2, hedge_costs, width, label='Hedge Costs',
                       color='lightcoral', alpha=0.7, edgecolor='black')

        ax3.set_ylabel('Cost ($)')
        ax3.set_title('Cost Breakdown')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=15)
        ax3.legend()
        ax3.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/risk_metrics.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved risk metrics plot to {self.save_dir}/risk_metrics.png")
        plt.close()

    def plot_return_distributions(self, figsize: tuple = (14, 6)):
        """
        Plot return distributions for each model

        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Return Distributions', fontsize=16, fontweight='bold')

        # 1. Distribution plots
        ax1 = axes[0]
        for model_name, result in self.results.items():
            if not result.pnl_series.empty:
                returns = result.pnl_series.diff().dropna()
                if len(returns) > 0:
                    ax1.hist(returns, bins=30, alpha=0.5, label=model_name.upper(),
                           edgecolor='black', density=True)

        ax1.set_xlabel('Return ($)')
        ax1.set_ylabel('Density')
        ax1.set_title('Return Distribution (Histogram)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # 2. Q-Q plots comparison
        ax2 = axes[1]
        for model_name, result in self.results.items():
            if not result.pnl_series.empty:
                returns = result.pnl_series.diff().dropna()
                if len(returns) > 1:
                    # Normalize returns
                    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-10)
                    returns_sorted = np.sort(returns_norm)

                    # Theoretical quantiles
                    n = len(returns_sorted)
                    theoretical_quantiles = np.linspace(-3, 3, n)

                    ax2.plot(theoretical_quantiles, returns_sorted,
                           'o', alpha=0.6, label=model_name.upper(), markersize=4)

        ax2.plot([-3, 3], [-3, 3], 'r--', alpha=0.5, linewidth=2, label='Normal')
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.set_title('Q-Q Plot (vs Normal Distribution)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/return_distributions.png", dpi=300, bbox_inches='tight')
        logger.info(f"Saved return distributions plot to {self.save_dir}/return_distributions.png")
        plt.close()

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create comprehensive summary table

        Returns:
        --------
        pd.DataFrame : Summary table
        """
        summary_data = []

        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name.upper(),
                'Total P&L ($)': f"{result.total_pnl:.2f}",
                'Gross P&L ($)': f"{result.gross_pnl:.2f}",
                'Net P&L ($)': f"{result.net_pnl:.2f}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.3f}",
                'Max Drawdown ($)': f"{result.max_drawdown:.2f}",
                'Win Rate (%)': f"{result.win_rate:.1f}",
                'Num Trades': result.num_trades,
                'Trans Costs ($)': f"{result.transaction_costs:.2f}",
                'Hedge Costs ($)': f"{result.hedge_costs:.2f}",
                'VaR 95% ($)': f"{result.var_95:.2f}"
            })

        df = pd.DataFrame(summary_data)

        # Save to CSV
        csv_path = f"{self.save_dir}/model_comparison_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary table to {csv_path}")

        return df

    def generate_report(self):
        """
        Generate comprehensive report with all visualizations
        """
        logger.info("=" * 60)
        logger.info("Generating Comprehensive Model Comparison Report")
        logger.info("=" * 60)

        # Create all plots
        self.plot_pnl_comparison()
        self.plot_risk_metrics()
        self.plot_return_distributions()

        # Create summary table
        summary_df = self.create_summary_table()

        # Print summary
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        print("=" * 60)

        # Generate markdown report
        self._generate_markdown_report(summary_df)

        logger.info("\nReport generation complete!")
        logger.info(f"All results saved to: {self.save_dir}/")

    def _generate_markdown_report(self, summary_df: pd.DataFrame):
        """Generate markdown report"""
        report_path = f"{self.save_dir}/model_comparison_report.md"

        with open(report_path, 'w') as f:
            f.write("# Bitcoin Options Trading Model Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This report compares the performance of multiple option pricing models ")
            f.write("(Heston, SABR, Local Volatility, Multi-Agent) on Bitcoin options trading.\n\n")

            f.write("## Performance Summary\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")

            f.write("## Key Findings\n\n")

            # Find best model by Sharpe ratio
            best_sharpe = summary_df.iloc[summary_df['Sharpe Ratio'].str.replace(r'[^\d.-]', '', regex=True).astype(float).idxmax()]
            f.write(f"- **Best Sharpe Ratio:** {best_sharpe['Model']} ({best_sharpe['Sharpe Ratio']})\n")

            # Find best model by P&L
            best_pnl = summary_df.iloc[summary_df['Total P&L ($)'].str.replace(r'[^\d.-]', '', regex=True).astype(float).idxmax()]
            f.write(f"- **Highest P&L:** {best_pnl['Model']} ({best_pnl['Total P&L ($)']})\n")

            # Find best win rate
            best_wr = summary_df.iloc[summary_df['Win Rate (%)'].str.replace(r'[^\d.-]', '', regex=True).astype(float).idxmax()]
            f.write(f"- **Best Win Rate:** {best_wr['Model']} ({best_wr['Win Rate (%)']})\\n")

            f.write("\n## Visualizations\n\n")
            f.write("![P&L Comparison](pnl_comparison.png)\n\n")
            f.write("![Risk Metrics](risk_metrics.png)\n\n")
            f.write("![Return Distributions](return_distributions.png)\n\n")

            f.write("## Conclusion\n\n")
            f.write("The Multi-Agent model aims to capture market microstructure effects ")
            f.write("that traditional models may miss, potentially leading to better trading performance. ")
            f.write("Compare the Sharpe ratios and P&L to determine if the added complexity is justified.\n")

        logger.info(f"Saved markdown report to {report_path}")


def quick_visualize(results: Dict, save_dir: str = "./results"):
    """
    Quick visualization function

    Parameters:
    -----------
    results : dict
        Dictionary of BacktestResult objects
    save_dir : str
        Save directory
    """
    visualizer = ModelComparisonVisualizer(results, save_dir)
    visualizer.generate_report()
