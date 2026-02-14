"""
Portfolio Optimizer - Dashboard Tables Service
===============================================

Generate comprehensive table data for the dashboard.
All tables needed for detailed analysis display.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DashboardTableService:
    """Generate all table data for the dashboard."""
    
    def get_performance_summary_table(self, metrics: Dict) -> Dict[str, Any]:
        """Generate performance summary table."""
        return {
            'title': 'Performance Summary',
            'headers': ['Metric', 'Value', 'Annualized'],
            'rows': [
                ['Total Return', f"{metrics.get('total_return', 0):.2%}", '-'],
                ['Annualized Return', f"{metrics.get('annualized_return', 0):.2%}", f"{metrics.get('annualized_return', 0):.2%}"],
                ['Volatility', f"{metrics.get('volatility', 0):.2%}", f"{metrics.get('volatility', 0):.2%}"],
                ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}", '-'],
                ['Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}", '-'],
                ['Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}", '-'],
                ['Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}", '-'],
                ['Max Drawdown Duration', f"{metrics.get('max_drawdown_days', 0)} days", '-'],
            ]
        }
    
    def get_risk_metrics_table(self, metrics: Dict, advanced_risk: Dict) -> Dict[str, Any]:
        """Generate comprehensive risk metrics table."""
        rows = [
            ['Value at Risk (95%)', f"{metrics.get('var_95', 0):.2%}", 'Historical'],
            ['Value at Risk (99%)', f"{metrics.get('var_99', 0):.2%}", 'Historical'],
            ['Conditional VaR (95%)', f"{metrics.get('cvar_95', 0):.2%}", 'Expected Shortfall'],
            ['Modified VaR (95%)', f"{advanced_risk.get('modified_var_95', 0):.2%}", 'Cornish-Fisher'],
            ['Beta', f"{metrics.get('beta', 0):.2f}", 'Market Sensitivity'],
            ['Alpha', f"{metrics.get('alpha', 0):.2%}", 'Annualized'],
            ['R-Squared', f"{metrics.get('r_squared', 0):.2%}", 'Systematic Risk'],
            ['Information Ratio', f"{metrics.get('information_ratio', 0):.2f}", '-'],
            ['Treynor Ratio', f"{metrics.get('treynor_ratio', 0):.2f}", '-'],
        ]
        
        # Add tail risk if available
        tail_risk = advanced_risk.get('tail_risk', {})
        if tail_risk:
            rows.extend([
                ['Tail Ratio', f"{tail_risk.get('tail_ratio_95_5', 0):.2f}", '95th/5th Percentile'],
                ['Kappa 3', f"{tail_risk.get('kappa_3', 0):.2f}", '3rd Order LPM'],
                ['Burke Ratio', f"{tail_risk.get('burke_ratio', 0):.2f}", '-'],
                ['Sterling Ratio', f"{tail_risk.get('sterling_ratio', 0):.2f}", '-'],
            ])
        
        return {
            'title': 'Risk Metrics',
            'headers': ['Metric', 'Value', 'Interpretation'],
            'rows': rows
        }
    
    def get_holdings_detail_table(self, holdings: List[Dict], 
                                  returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed holdings table."""
        rows = []
        
        for holding in holdings:
            ticker = holding['ticker']
            weight = holding.get('weight', 0)
            sector = holding.get('sector', 'Unknown')
            
            # Get individual stock metrics if available
            if ticker in returns_df.columns:
                stock_returns = returns_df[ticker]
                ann_return = stock_returns.mean() * 252
                ann_vol = stock_returns.std() * np.sqrt(252)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            else:
                ann_return = ann_vol = sharpe = 0
            
            rows.append([
                ticker,
                sector,
                f"{weight:.2%}",
                f"{ann_return:.2%}",
                f"{ann_vol:.2%}",
                f"{sharpe:.2f}"
            ])
        
        # Sort by weight descending
        rows.sort(key=lambda x: float(x[2].rstrip('%')), reverse=True)
        
        return {
            'title': 'Holdings Detail',
            'headers': ['Ticker', 'Sector', 'Weight', 'Ann. Return', 'Ann. Vol', 'Sharpe'],
            'rows': rows
        }
    
    def get_sector_allocation_table(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Generate sector allocation breakdown."""
        sector_weights = {}
        sector_counts = {}
        
        for holding in holdings:
            sector = holding.get('sector', 'Unknown')
            weight = holding.get('weight', 0)
            
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        rows = []
        for sector, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
            rows.append([
                sector,
                f"{weight:.2%}",
                sector_counts[sector],
                'Overweight' if weight > 0.25 else 'Normal' if weight > 0.15 else 'Underweight'
            ])
        
        return {
            'title': 'Sector Allocation',
            'headers': ['Sector', 'Weight', '# Stocks', 'Status'],
            'rows': rows
        }
    
    def get_factor_exposure_table(self, factor_exposure: Dict) -> Dict[str, Any]:
        """Generate factor exposure table."""
        betas = factor_exposure.get('betas', {})
        contributions = factor_exposure.get('factor_contributions', {})
        
        rows = [
            ['Market (MKT-RF)', f"{betas.get('market', 0):.2f}", f"{contributions.get('market', 0):.2%}", 'Market Risk'],
            ['Size (SMB)', f"{betas.get('smb', 0):.2f}", f"{contributions.get('size', 0):.2%}", 'Small Cap Premium'],
            ['Value (HML)', f"{betas.get('hml', 0):.2f}", f"{contributions.get('value', 0):.2%}", 'Value Premium'],
            ['Profitability (RMW)', f"{betas.get('rmw', 0):.2f}", f"{contributions.get('profitability', 0):.2%}", 'Quality Factor'],
            ['Investment (CMA)', f"{betas.get('cma', 0):.2f}", f"{contributions.get('investment', 0):.2%}", 'Conservative Investment'],
            ['Momentum (MOM)', f"{betas.get('mom', 0):.2f}", f"{contributions.get('momentum', 0):.2%}", 'Momentum Factor'],
        ]
        
        return {
            'title': 'Factor Exposure (Fama-French 5-Factor)',
            'headers': ['Factor', 'Beta', 'Contribution', 'Description'],
            'rows': rows,
            'summary': {
                'alpha': f"{factor_exposure.get('alpha', 0):.2%}",
                'r_squared': f"{factor_exposure.get('r_squared', 0):.2%}"
            }
        }
    
    def get_monte_carlo_summary_table(self, mc_result: Dict) -> Dict[str, Any]:
        """Generate Monte Carlo simulation summary."""
        return {
            'title': 'Monte Carlo Simulation (5,000 Paths, 1 Year)',
            'headers': ['Scenario', 'Portfolio Value', 'Return'],
            'rows': [
                ['Initial Investment', f"${mc_result.get('initial_value', 100000):,.0f}", '-'],
                ['Best Case (95th %ile)', f"${mc_result.get('best_case', 0):,.0f}", 
                 f"{(mc_result.get('best_case', 0) / 100000 - 1):.1%}"],
                ['Expected (Mean)', f"${mc_result.get('mean_final', 0):,.0f}",
                 f"{(mc_result.get('mean_final', 0) / 100000 - 1):.1%}"],
                ['Median', f"${mc_result.get('median_final', 0):,.0f}",
                 f"{(mc_result.get('median_final', 0) / 100000 - 1):.1%}"],
                ['Worst Case (5th %ile)', f"${mc_result.get('worst_case', 0):,.0f}",
                 f"{(mc_result.get('worst_case', 0) / 100000 - 1):.1%}"],
            ],
            'probabilities': {
                'profit': f"{mc_result.get('probability_profit', 0):.1%}",
                'loss': f"{mc_result.get('probability_loss', 0):.1%}",
                'target_10pct': f"{mc_result.get('probability_target', 0):.1%}"
            }
        }
    
    def get_stress_test_table(self, stress_results: Dict) -> Dict[str, Any]:
        """Generate stress test results table."""
        rows = []
        for scenario, impact in sorted(stress_results.items(), key=lambda x: x[1]):
            severity = 'Severe' if impact < -0.30 else 'High' if impact < -0.20 else 'Moderate' if impact < -0.10 else 'Low'
            rows.append([
                scenario,
                f"{impact:.1%}",
                severity,
                'Critical' if impact < -0.30 else 'Warning' if impact < -0.15 else 'Acceptable'
            ])
        
        return {
            'title': 'Stress Test Scenarios',
            'headers': ['Scenario', 'Estimated Impact', 'Severity', 'Status'],
            'rows': rows
        }
    
    def get_optimization_comparison_table(self, optimizations: Dict) -> Dict[str, Any]:
        """Generate optimization methods comparison."""
        current = optimizations.get('current', {})
        max_sharpe = optimizations.get('max_sharpe', {})
        min_var = optimizations.get('min_variance', {})
        risk_parity = optimizations.get('risk_parity', {})
        
        rows = [
            ['Current Portfolio',
             f"{current.get('expected_return', 0):.2%}",
             f"{current.get('volatility', 0):.2%}",
             f"{current.get('sharpe_ratio', 0):.2f}"],
            ['Max Sharpe Ratio',
             f"{max_sharpe.get('expected_return', 0):.2%}",
             f"{max_sharpe.get('volatility', 0):.2%}",
             f"{max_sharpe.get('sharpe_ratio', 0):.2f}"],
            ['Minimum Variance',
             f"{min_var.get('expected_return', 0):.2%}",
             f"{min_var.get('volatility', 0):.2%}",
             f"{min_var.get('sharpe_ratio', 0):.2f}"],
            ['Risk Parity',
             f"{risk_parity.get('expected_return', 0):.2%}",
             f"{risk_parity.get('volatility', 0):.2%}",
             f"{risk_parity.get('sharpe_ratio', 0):.2f}"],
        ]
        
        return {
            'title': 'Optimization Comparison',
            'headers': ['Method', 'Expected Return', 'Volatility', 'Sharpe Ratio'],
            'rows': rows
        }
    
    def get_statistical_tests_table(self, stat_tests: Dict) -> Dict[str, Any]:
        """Generate statistical tests results table."""
        return {
            'title': 'Statistical Tests',
            'headers': ['Test', 'Statistic', 'P-Value', 'Result'],
            'rows': [
                ['Jarque-Bera Normality',
                 f"{stat_tests.get('jarque_bera_stat', 0):.2f}",
                 f"{stat_tests.get('jarque_bera_pvalue', 0):.4f}",
                 'Normal' if stat_tests.get('is_normal') else 'Non-Normal'],
                ['Ljung-Box Autocorr.',
                 f"{stat_tests.get('ljung_box_stat', 0):.2f}",
                 f"{stat_tests.get('ljung_box_pvalue', 0):.4f}",
                 'No Autocorr.' if not stat_tests.get('has_autocorrelation') else 'Autocorrelation Detected'],
            ]
        }
    
    def get_correlation_matrix_table(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation matrix as table."""
        corr = returns_df.corr()
        tickers = corr.columns.tolist()
        
        return {
            'title': 'Correlation Matrix',
            'tickers': tickers,
            'matrix': corr.values.tolist(),
            'average_corr': corr.values[np.triu_indices_from(corr.values, k=1)].mean()
        }
    
    def get_all_tables(self, analysis_data: Dict) -> Dict[str, Any]:
        """Generate all tables at once."""
        return {
            'performance_summary': self.get_performance_summary_table(
                analysis_data.get('metrics', {})
            ),
            'risk_metrics': self.get_risk_metrics_table(
                analysis_data.get('metrics', {}),
                analysis_data.get('advanced_risk', {})
            ),
            'holdings': self.get_holdings_detail_table(
                analysis_data.get('holdings', []),
                pd.DataFrame()  # Would need returns_df passed in
            ),
            'sector_allocation': self.get_sector_allocation_table(
                analysis_data.get('holdings', [])
            ),
            'factor_exposure': self.get_factor_exposure_table(
                analysis_data.get('factor_exposure', {})
            ),
            'monte_carlo': self.get_monte_carlo_summary_table(
                analysis_data.get('monte_carlo', {})
            ),
            'stress_test': self.get_stress_test_table(
                analysis_data.get('stress_test', {})
            ),
            'optimization': self.get_optimization_comparison_table(
                analysis_data.get('optimizations', {})
            ),
            'statistical_tests': self.get_statistical_tests_table(
                analysis_data.get('statistical_tests', {})
            )
        }


# Singleton instance
dashboard_tables = DashboardTableService()
