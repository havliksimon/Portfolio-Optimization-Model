"""
Portfolio Optimizer - Core Optimization Engine
==============================================

Implementation of Modern Portfolio Theory (MPT) and related quantitative
methods for optimal asset allocation. This module implements the mathematical
frameworks developed by Markowitz, Sharpe, and subsequent researchers.

Mathematical Framework:
-----------------------
Given n assets with expected returns μ ∈ R^n and covariance Σ ∈ R^{n×n},
find optimal weights w* that maximize risk-adjusted return:

    w* = argmax_w { (w^T μ - r_f) / √(w^T Σ w) }

Subject to constraints:
    1^T w = 1      (fully invested)
    w ≥ 0          (no short sales, optional)
    w_min ≤ w ≤ w_max  (position limits)

Optimization Methods:
--------------------
1. Mean-Variance Optimization (Markowitz, 1952)
2. Maximum Sharpe Ratio (Tobin, 1958)
3. Minimum Variance Portfolio
4. Risk Parity (Maillard, Roncalli & Teïletche, 2010)
5. Black-Litterman (Black & Litterman, 1992)

References:
-----------
- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
- Sharpe, W. F. (1966). Mutual Fund Performance. The Journal of Business, 39(1), 119-138.
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization. Financial Analysts Journal.
- Roncalli, T. (2013). Introduction to Risk Parity and Budgeting. CRC Press.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from services.market_data import PriceData

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Enumeration of supported optimization methodologies."""
    MEAN_VARIANCE = "mean_variance"           # Classic Markowitz
    MAX_SHARPE = "max_sharpe"                 # Tangency portfolio
    MIN_VARIANCE = "min_variance"             # Global minimum variance
    RISK_PARITY = "risk_parity"               # Equal risk contribution
    HIERARCHICAL_RISK = "hierarchical_risk"   # Hierarchical Risk Parity (De Prado, 2016)
    BLACK_LITTERMAN = "black_litterman"       # Bayesian approach


@dataclass
class OptimizationConstraints:
    """
    Constraint specification for portfolio optimization.
    
    Implements the Specification pattern allowing flexible constraint
    configuration while maintaining mathematical validity.
    
    Attributes:
        allow_short: Whether negative weights are permitted
        max_position_size: Maximum weight for any single asset (0-1)
        min_position_size: Minimum weight if asset is held
        target_return: Required return for return-target optimization
        target_volatility: Required volatility for risk-target optimization
        sector_constraints: Dict mapping sector to (min, max) tuple
    """
    allow_short: bool = False
    max_position_size: float = 1.0
    min_position_size: float = 0.0
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    sector_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Ensure constraint consistency."""
        if self.max_position_size > 1.0 or self.min_position_size < 0:
            return False
        if not self.allow_short and self.min_position_size < 0:
            return False
        return True


@dataclass
class OptimizationResult:
    """
    Immutable result container for optimization execution.
    
    Attributes:
        weights: Optimal asset allocation
        expected_return: Annualized expected return
        volatility: Annualized portfolio volatility
        sharpe_ratio: Risk-adjusted return metric
        method: Optimization algorithm used
        metrics: Additional performance metrics
    """
    weights: np.ndarray
    tickers: List[str]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str
    risk_free_rate: float = 0.05
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'weights': {t: float(w) for t, w in zip(self.tickers, self.weights)},
            'expected_return': self.expected_return,
            'expected_return_pct': f"{self.expected_return:.2%}",
            'volatility': self.volatility,
            'volatility_pct': f"{self.volatility:.2%}",
            'sharpe_ratio': self.sharpe_ratio,
            'method': self.method,
            'risk_free_rate': self.risk_free_rate,
            'metrics': self.metrics,
        }
    
    @property
    def diversification_ratio(self) -> float:
        """Portfolio diversification ratio (weighted avg vol / portfolio vol)."""
        if 'asset_vols' in self.metrics:
            weighted_avg = np.dot(self.weights, self.metrics['asset_vols'])
            return weighted_avg / self.volatility if self.volatility > 0 else 1.0
        return 1.0
    
    @property
    def concentration(self) -> float:
        """Herfindahl-Hirschman Index of concentration."""
        return float(np.sum(self.weights ** 2))


class PortfolioOptimizer:
    """
    Core optimization engine implementing multiple portfolio construction methods.
    
    This class implements the Strategy pattern, allowing runtime selection
    of optimization algorithms while maintaining consistent interfaces.
    
    Usage:
    ------
        optimizer = PortfolioOptimizer(returns_df, risk_free_rate=0.05)
        result = optimizer.optimize(method=OptimizationMethod.MAX_SHARPE)
    
    Mathematical Notation:
    ----------------------
    - μ: Vector of expected returns
    - Σ: Covariance matrix of returns
    - w: Portfolio weight vector
    - σ_p: Portfolio volatility = √(w^T Σ w)
    """
    
    def __init__(
        self,
        returns_df: pd.DataFrame,
        risk_free_rate: float = 0.05,
        freq: int = 252
    ):
        """
        Initialize optimizer with historical return data.
        
        Args:
            returns_df: DataFrame of asset returns (dates × assets)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            freq: Trading days per year for annualization
        """
        self.returns = returns_df.dropna()
        self.tickers = list(self.returns.columns)
        self.n_assets = len(self.tickers)
        self.risk_free_rate = risk_free_rate
        self.freq = freq
        
        # Calculate expected returns and covariance
        self.mu = self.returns.mean() * freq  # Annualized expected returns
        self.Sigma = self.returns.cov() * freq  # Annualized covariance
        
        # Validate inputs
        if self.n_assets == 0:
            raise ValueError("No valid return data provided")
        
        logger.info(f"Optimizer initialized with {self.n_assets} assets")
    
    def optimize(
        self,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        constraints: Optional[OptimizationConstraints] = None,
        initial_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Execute portfolio optimization with specified method.
        
        Args:
            method: Optimization algorithm to use
            constraints: Portfolio constraints
            initial_weights: Starting point for optimization
            
        Returns:
            OptimizationResult with optimal allocation
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        if initial_weights is None:
            initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Dispatch to appropriate method
        method_map = {
            OptimizationMethod.MEAN_VARIANCE: self._mean_variance_opt,
            OptimizationMethod.MAX_SHARPE: self._max_sharpe_opt,
            OptimizationMethod.MIN_VARIANCE: self._min_variance_opt,
            OptimizationMethod.RISK_PARITY: self._risk_parity_opt,
        }
        
        opt_func = method_map.get(method)
        if opt_func is None:
            raise NotImplementedError(f"Method {method} not yet implemented")
        
        return opt_func(constraints, initial_weights)
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility given weights."""
        return np.sqrt(np.dot(weights.T, np.dot(self.Sigma.values, weights)))
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio expected return given weights."""
        return np.dot(weights, self.mu.values)
    
    def _sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio given weights."""
        p_return = self._portfolio_return(weights)
        p_vol = self._portfolio_volatility(weights)
        if p_vol == 0:
            return -np.inf
        return (p_return - self.risk_free_rate) / p_vol
    
    def _get_bounds(self, constraints: OptimizationConstraints) -> List[Tuple[float, float]]:
        """Generate weight bounds from constraints."""
        if constraints.allow_short:
            return [(-constraints.max_position_size, constraints.max_position_size)] * self.n_assets
        return [(0, constraints.max_position_size)] * self.n_assets
    
    def _max_sharpe_opt(
        self,
        constraints: OptimizationConstraints,
        initial_weights: np.ndarray
    ) -> OptimizationResult:
        """
        Maximum Sharpe Ratio Optimization (Tobin, 1958).
        
        Finds the tangency portfolio on the efficient frontier that
        maximizes the reward-to-variability ratio.
        
        This is equivalent to:
            max_w (μ^T w - r_f) / √(w^T Σ w)
            s.t. 1^T w = 1, w ≥ 0
        """
        
        def neg_sharpe(weights):
            return -self._sharpe_ratio(weights)
        
        # Constraints
        bounds = self._get_bounds(constraints)
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Solve
        result = minimize(
            neg_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        optimal_weights = result.x
        opt_return = self._portfolio_return(optimal_weights)
        opt_vol = self._portfolio_volatility(optimal_weights)
        opt_sharpe = self._sharpe_ratio(optimal_weights)
        
        return OptimizationResult(
            weights=optimal_weights,
            tickers=self.tickers,
            expected_return=opt_return,
            volatility=opt_vol,
            sharpe_ratio=opt_sharpe,
            method=OptimizationMethod.MAX_SHARPE.value,
            risk_free_rate=self.risk_free_rate,
            metrics={
                'success': result.success,
                'iterations': result.nit,
                'asset_vols': [float(self.Sigma.iloc[i, i] ** 0.5) for i in range(self.n_assets)]
            }
        )
    
    def _min_variance_opt(
        self,
        constraints: OptimizationConstraints,
        initial_weights: np.ndarray
    ) -> OptimizationResult:
        """
        Global Minimum Variance Portfolio.
        
        Finds the portfolio with lowest possible volatility without
        regard to expected returns. Useful for risk-averse investors.
        
        min_w w^T Σ w
        s.t. 1^T w = 1, w ≥ 0
        """
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.Sigma.values, weights))
        
        bounds = self._get_bounds(constraints)
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        optimal_weights = result.x
        opt_return = self._portfolio_return(optimal_weights)
        opt_vol = self._portfolio_volatility(optimal_weights)
        opt_sharpe = self._sharpe_ratio(optimal_weights)
        
        return OptimizationResult(
            weights=optimal_weights,
            tickers=self.tickers,
            expected_return=opt_return,
            volatility=opt_vol,
            sharpe_ratio=opt_sharpe,
            method=OptimizationMethod.MIN_VARIANCE.value,
            risk_free_rate=self.risk_free_rate,
            metrics={'success': result.success}
        )
    
    def _mean_variance_opt(
        self,
        constraints: OptimizationConstraints,
        initial_weights: np.ndarray
    ) -> OptimizationResult:
        """
        Mean-Variance Optimization with target return constraint.
        
        min_w w^T Σ w
        s.t. μ^T w = target_return, 1^T w = 1, w ≥ 0
        """
        target = constraints.target_return or self.mu.mean()
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.Sigma.values, weights))
        
        bounds = self._get_bounds(constraints)
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self._portfolio_return(x) - target}
        ]
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        optimal_weights = result.x
        opt_return = self._portfolio_return(optimal_weights)
        opt_vol = self._portfolio_volatility(optimal_weights)
        opt_sharpe = self._sharpe_ratio(optimal_weights)
        
        return OptimizationResult(
            weights=optimal_weights,
            tickers=self.tickers,
            expected_return=opt_return,
            volatility=opt_vol,
            sharpe_ratio=opt_sharpe,
            method=OptimizationMethod.MEAN_VARIANCE.value,
            risk_free_rate=self.risk_free_rate,
            metrics={'target_return': target, 'success': result.success}
        )
    
    def _risk_parity_opt(
        self,
        constraints: OptimizationConstraints,
        initial_weights: np.ndarray
    ) -> OptimizationResult:
        """
        Risk Parity / Equal Risk Contribution Portfolio.
        
        Allocates such that each asset contributes equally to portfolio risk.
        This approach is advocated by Qian (2005) and others as an alternative
        to equal-weighting that accounts for risk.
        
        min_w Σ_i Σ_j (w_i(Σw)_i - w_j(Σw)_j)^2
        s.t. 1^T w = 1, w ≥ 0
        """
        
        def risk_parity_objective(weights):
            portfolio_vol = self._portfolio_volatility(weights)
            if portfolio_vol == 0:
                return np.inf
            
            # Marginal risk contribution
            marginal_risk = np.dot(self.Sigma.values, weights) / portfolio_vol
            
            # Risk contribution per asset
            risk_contrib = weights * marginal_risk
            
            # Target: equal contribution (1/n each)
            target_contrib = portfolio_vol / self.n_assets
            
            # Minimize squared deviation from target
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        bounds = self._get_bounds(constraints)
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        optimal_weights = result.x
        opt_return = self._portfolio_return(optimal_weights)
        opt_vol = self._portfolio_volatility(optimal_weights)
        opt_sharpe = self._sharpe_ratio(optimal_weights)
        
        # Calculate risk contributions for metrics
        marginal_risk = np.dot(self.Sigma.values, optimal_weights) / opt_vol
        risk_contribs = optimal_weights * marginal_risk
        
        return OptimizationResult(
            weights=optimal_weights,
            tickers=self.tickers,
            expected_return=opt_return,
            volatility=opt_vol,
            sharpe_ratio=opt_sharpe,
            method=OptimizationMethod.RISK_PARITY.value,
            risk_free_rate=self.risk_free_rate,
            metrics={
                'risk_contributions': risk_contribs.tolist(),
                'success': result.success
            }
        )
    
    def efficient_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[OptimizationConstraints] = None
    ) -> pd.DataFrame:
        """
        Generate the efficient frontier by varying target returns.
        
        The efficient frontier represents the set of optimal portfolios
        offering the highest expected return for a defined level of risk.
        
        Args:
            n_points: Number of portfolio points to calculate
            constraints: Portfolio constraints
            
        Returns:
            DataFrame with columns [return, volatility, sharpe, weights...]
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Determine return range
        min_return = self.mu.min()
        max_return = self.mu.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        results = []
        for target in target_returns:
            try:
                constraints_copy = OptimizationConstraints(
                    allow_short=constraints.allow_short,
                    max_position_size=constraints.max_position_size,
                    target_return=target
                )
                result = self._mean_variance_opt(constraints_copy, 
                                                 np.ones(self.n_assets) / self.n_assets)
                results.append({
                    'return': result.expected_return,
                    'volatility': result.volatility,
                    'sharpe': result.sharpe_ratio,
                    **{f'weight_{t}': w for t, w in zip(result.tickers, result.weights)}
                })
            except Exception as e:
                logger.warning(f"Failed to optimize for target {target}: {e}")
        
        return pd.DataFrame(results)
