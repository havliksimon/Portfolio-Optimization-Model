"""
Portfolio Optimizer - Example Portfolio Service
================================================

Pre-computed example portfolio for visitors.
Runs advanced analysis and caches results for fast display.
"""

import json
import pickle
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import logging

import pandas as pd
import numpy as np

from services.market_data import market_data_service
from services.risk_analytics import risk_analytics
from services.advanced_statistics import advanced_statistics
from services.advanced_risk import AdvancedRiskCalculator
from services.factor_models import FamaFrenchModel
from services.black_litterman import BlackLittermanModel
from services.dashboard_charts import dashboard_charts
from config import config

logger = logging.getLogger(__name__)


# Example portfolio - Berkshire-inspired diversified portfolio
EXAMPLE_PORTFOLIO = {
    'name': 'Example Diversified Portfolio',
    'description': 'A balanced portfolio inspired by value investing principles with tech exposure',
    'holdings': [
        {'ticker': 'AAPL', 'weight': 0.15, 'sector': 'Technology'},
        {'ticker': 'MSFT', 'weight': 0.15, 'sector': 'Technology'},
        {'ticker': 'GOOGL', 'weight': 0.12, 'sector': 'Technology'},
        {'ticker': 'AMZN', 'weight': 0.10, 'sector': 'Consumer Discretionary'},
        {'ticker': 'NVDA', 'weight': 0.08, 'sector': 'Technology'},
        {'ticker': 'JPM', 'weight': 0.08, 'sector': 'Financials'},
        {'ticker': 'JNJ', 'weight': 0.07, 'sector': 'Healthcare'},
        {'ticker': 'V', 'weight': 0.07, 'sector': 'Financials'},
        {'ticker': 'PG', 'weight': 0.06, 'sector': 'Consumer Staples'},
        {'ticker': 'UNH', 'weight': 0.06, 'sector': 'Healthcare'},
        {'ticker': 'HD', 'weight': 0.04, 'sector': 'Consumer Discretionary'},
        {'ticker': 'MA', 'weight': 0.04, 'sector': 'Financials'},
        {'ticker': 'LLY', 'weight': 0.04, 'sector': 'Healthcare'},
        {'ticker': 'KO', 'weight': 0.03, 'sector': 'Consumer Staples'},
        {'ticker': 'PEP', 'weight': 0.01, 'sector': 'Consumer Staples'}
    ],
    'period': '2y'
}


class ExamplePortfolioService:
    """
    Service for managing and serving the example portfolio.
    
    Pre-computes all analysis and caches results.
    Updates daily to keep data fresh.
    """
    
    def __init__(self):
        self._cache = {}
        self._last_update = None
        self._cache_duration = timedelta(hours=6)  # Update every 6 hours
        
    def _should_refresh(self) -> bool:
        """Check if cached data needs refresh."""
        if self._last_update is None:
            return True
        return datetime.utcnow() - self._last_update > self._cache_duration
    
    def _compute_portfolio_analysis(self) -> Dict[str, Any]:
        """
        Compute comprehensive portfolio analysis.
        
        This is the heavy lifting - fetches data and runs all analyses.
        """
        logger.info("Computing example portfolio analysis...")
        
        holdings = EXAMPLE_PORTFOLIO['holdings']
        tickers = [h['ticker'] for h in holdings]
        weights_dict = {h['ticker']: h['weight'] for h in holdings}
        period = EXAMPLE_PORTFOLIO['period']
        
        # Fetch market data
        batch_data = market_data_service.fetch_batch_data(tickers, period=period)
        valid_data = {t: d for t, d in batch_data.items() if d is not None}
        
        if len(valid_data) < 5:
            logger.error(f"Insufficient data for example portfolio: {len(valid_data)} tickers")
            return self._get_fallback_data()
        
        # Build returns DataFrame
        returns_df = pd.DataFrame({t: d.returns for t, d in valid_data.items()}).dropna()
        
        # Portfolio returns
        weights_array = np.array([weights_dict.get(t, 0) for t in returns_df.columns])
        weights_array = weights_array / weights_array.sum()  # Normalize
        portfolio_returns = returns_df @ weights_array
        
        # 1. Basic Risk Metrics
        logger.info("Computing risk metrics...")
        spy_data = market_data_service.fetch_historical_data('SPY', period=period)
        benchmark_returns = spy_data.returns if spy_data else None
        
        metrics = risk_analytics.calculate_comprehensive_metrics(
            portfolio_returns,
            benchmark_returns=benchmark_returns
        )
        
        # 2. Rolling Metrics
        logger.info("Computing rolling metrics...")
        rolling = risk_analytics.calculate_rolling_metrics(
            portfolio_returns,
            benchmark_returns=benchmark_returns
        )
        
        # 3. Advanced Risk
        logger.info("Computing advanced risk...")
        adv_calc = AdvancedRiskCalculator(returns_df, weights_array)
        advanced_risk = {
            'modified_var_95': adv_calc.modified_var(0.95),
            'modified_var_99': adv_calc.modified_var(0.99),
            'kelly': adv_calc.kelly_criterion(),
            'ulcer': adv_calc.ulcer_index(),
            'drawdown_risk': adv_calc.drawdown_at_risk(),
            'tail_risk': adv_calc.tail_risk_metrics()
        }
        
        # 4. Distribution Fitting
        logger.info("Fitting distributions...")
        distributions = advanced_statistics.fit_distributions(portfolio_returns)
        
        # 5. Monte Carlo
        logger.info("Running Monte Carlo...")
        mc_result = advanced_statistics.monte_carlo_simulation(
            portfolio_returns,
            weights_array,
            initial_value=100000,
            n_simulations=5000,
            n_days=252
        )
        
        # 6. Statistical Tests
        logger.info("Running statistical tests...")
        stat_tests = advanced_statistics.run_statistical_tests(portfolio_returns)
        
        # 7. Factor Exposure
        logger.info("Computing factor exposure...")
        try:
            ff_model = FamaFrenchModel()
            factor_exposure = ff_model.estimate_exposure(portfolio_returns, model='5factor')
            factor_data = {
                'market_beta': factor_exposure.market_beta,
                'smb_beta': factor_exposure.smb_beta,
                'hml_beta': factor_exposure.hml_beta,
                'rmw_beta': factor_exposure.rmw_beta,
                'cma_beta': factor_exposure.cma_beta,
                'alpha': factor_exposure.alpha,
                'r_squared': factor_exposure.r_squared,
                'factor_contributions': factor_exposure.factor_contributions
            }
        except Exception as e:
            logger.warning(f"Factor exposure failed: {e}")
            factor_data = {}
        
        # 8. PCA
        logger.info("Computing PCA...")
        pca_result = advanced_statistics.principal_component_analysis(returns_df)
        
        # 9. Volatility Analysis
        logger.info("Analyzing volatility...")
        vol_analysis = advanced_statistics.analyze_volatility_clustering(portfolio_returns)
        
        # 10. Tail Risk
        logger.info("Computing tail risk...")
        tail_risk = advanced_statistics.extreme_value_analysis(portfolio_returns)
        
        # 11. Stress Test
        logger.info("Running stress tests...")
        stress_results = risk_analytics.stress_test(portfolio_returns, weights_dict)
        
        # 12. All Charts
        logger.info("Generating charts...")
        charts = dashboard_charts.get_all_charts(
            returns_df,
            portfolio_returns,
            weights_dict,
            factor_data,
            stress_results,
            period
        )
        
        # 13. Optimization Alternatives
        logger.info("Computing optimizations...")
        from services.optimization import PortfolioOptimizer, OptimizationConstraints, OptimizationMethod
        
        optimizer = PortfolioOptimizer(returns_df)
        constraints = OptimizationConstraints(max_position_size=0.20)
        
        optimizations = {}
        try:
            max_sharpe = optimizer.optimize(method=OptimizationMethod.MAX_SHARPE, constraints=constraints)
            optimizations['max_sharpe'] = {
                'weights': {returns_df.columns[i]: max_sharpe.weights[i] 
                           for i in range(len(returns_df.columns))},
                'expected_return': max_sharpe.expected_return,
                'volatility': max_sharpe.volatility,
                'sharpe_ratio': max_sharpe.sharpe_ratio
            }
        except Exception as e:
            logger.warning(f"Max Sharpe failed: {e}")
        
        try:
            min_var = optimizer.optimize(method=OptimizationMethod.MIN_VARIANCE, constraints=constraints)
            optimizations['min_variance'] = {
                'weights': {returns_df.columns[i]: min_var.weights[i] 
                           for i in range(len(returns_df.columns))},
                'expected_return': min_var.expected_return,
                'volatility': min_var.volatility,
                'sharpe_ratio': min_var.sharpe_ratio
            }
        except Exception as e:
            logger.warning(f"Min Variance failed: {e}")
        
        # Compile results
        results = {
            'meta': {
                'name': EXAMPLE_PORTFOLIO['name'],
                'description': EXAMPLE_PORTFOLIO['description'],
                'computed_at': datetime.utcnow().isoformat(),
                'period': period,
                'n_assets': len(valid_data)
            },
            'holdings': [
                {
                    'ticker': h['ticker'],
                    'weight': float(weights_dict.get(h['ticker'], 0)),
                    'sector': h.get('sector', 'Unknown')
                }
                for h in holdings if h['ticker'] in valid_data
            ],
            'metrics': {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'beta': metrics.beta,
                'alpha': metrics.alpha
            },
            'advanced_risk': advanced_risk,
            'distributions': {
                name: {
                    'mean': dist.mean,
                    'std': dist.std,
                    'aic': dist.aic,
                    'bic': dist.bic,
                    'p_value': dist.p_value
                }
                for name, dist in distributions.items()
            },
            'monte_carlo': {
                'mean_final': mc_result.mean_final,
                'median_final': mc_result.median_final,
                'var_95': mc_result.var_95,
                'probability_profit': mc_result.probability_profit,
                'paths': mc_result.paths[::100].tolist() if len(mc_result.paths) > 0 else []
            },
            'statistical_tests': {
                'jarque_bera_stat': stat_tests.jarque_bera_stat,
                'jarque_bera_pvalue': stat_tests.jarque_bera_pvalue,
                'is_normal': stat_tests.is_normal,
                'ljung_box_stat': stat_tests.ljung_box_stat,
                'has_autocorrelation': stat_tests.has_autocorrelation
            },
            'factor_exposure': factor_data,
            'pca': {
                'explained_variance_ratio': pca_result.explained_variance_ratio[:5].tolist(),
                'condition_number': pca_result.condition_number
            },
            'volatility_analysis': {
                'has_arch_effects': vol_analysis.has_arch_effects,
                'arch_lm_pvalue': vol_analysis.arch_lm_pvalue
            },
            'tail_risk': {
                'hill_estimator': tail_risk.hill_estimator,
                'tail_index': tail_risk.tail_index,
                'black_swan_prob': tail_risk.black_swan_prob
            },
            'stress_test': stress_results,
            'optimizations': optimizations,
            'charts': charts
        }
        
        logger.info("Example portfolio analysis complete!")
        return results
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Return fallback data if analysis fails."""
        return {
            'meta': {
                'name': EXAMPLE_PORTFOLIO['name'],
                'error': 'Data temporarily unavailable',
                'computed_at': datetime.utcnow().isoformat()
            },
            'holdings': EXAMPLE_PORTFOLIO['holdings'],
            'metrics': {}
        }
    
    def get_analysis(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get example portfolio analysis.
        
        Returns cached data if available and fresh,
        otherwise computes new analysis.
        """
        cache_key = 'example_portfolio_analysis'
        
        if not force_refresh and not self._should_refresh():
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Compute fresh analysis
        results = self._compute_portfolio_analysis()
        
        # Cache results
        self._cache[cache_key] = results
        self._last_update = datetime.utcnow()
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary for display."""
        analysis = self.get_analysis()
        
        return {
            'name': analysis['meta']['name'],
            'description': analysis['meta']['description'],
            'holdings': analysis['holdings'][:10],  # Top 10
            'key_metrics': {
                'annualized_return': analysis.get('metrics', {}).get('annualized_return', 0),
                'volatility': analysis.get('metrics', {}).get('volatility', 0),
                'sharpe_ratio': analysis.get('metrics', {}).get('sharpe_ratio', 0),
                'max_drawdown': analysis.get('metrics', {}).get('max_drawdown', 0)
            },
            'computed_at': analysis['meta']['computed_at']
        }


# Singleton instance
example_portfolio_service = ExamplePortfolioService()
