"""
Portfolio Optimizer - Optimization Engine Tests
===============================================

Unit tests for the portfolio optimization engine covering:
- Mean-variance optimization
- Maximum Sharpe ratio calculation
- Constraint handling
- Efficient frontier generation

Test Philosophy:
----------------
- Property-based testing for mathematical invariants
- Edge case coverage for constraint violations
- Numerical stability validation
"""

import pytest
import numpy as np
import pandas as pd
from services.optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationConstraints,
    OptimizationResult
)


@pytest.fixture
def sample_returns():
    """Generate sample return data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='B')
    n_assets = 5
    
    # Generate correlated returns
    mean = np.array([0.0005, 0.0004, 0.0003, 0.0006, 0.0004])
    cov = np.array([
        [0.0004, 0.0001, 0.0001, 0.0001, 0.0001],
        [0.0001, 0.0003, 0.0001, 0.0001, 0.0001],
        [0.0001, 0.0001, 0.0002, 0.0001, 0.0001],
        [0.0001, 0.0001, 0.0001, 0.0005, 0.0001],
        [0.0001, 0.0001, 0.0001, 0.0001, 0.0003]
    ])
    
    returns = np.random.multivariate_normal(mean, cov, len(dates))
    return pd.DataFrame(
        returns,
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    )


@pytest.fixture
def optimizer(sample_returns):
    """Create optimizer instance."""
    return PortfolioOptimizer(sample_returns, risk_free_rate=0.05)


class TestPortfolioOptimizer:
    """Test suite for PortfolioOptimizer class."""
    
    def test_initialization(self, sample_returns):
        """Test optimizer initialization."""
        opt = PortfolioOptimizer(sample_returns)
        
        assert opt.n_assets == 5
        assert len(opt.tickers) == 5
        assert opt.mu.shape[0] == 5
        assert opt.Sigma.shape == (5, 5)
    
    def test_max_sharpe_optimization(self, optimizer):
        """Test maximum Sharpe ratio optimization."""
        result = optimizer.optimize(method=OptimizationMethod.MAX_SHARPE)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 5
        assert abs(sum(result.weights) - 1.0) < 1e-6  # Fully invested
        assert all(w >= 0 for w in result.weights)  # No short sales
        assert result.sharpe_ratio > 0
        assert result.expected_return > 0
        assert result.volatility > 0
    
    def test_min_variance_optimization(self, optimizer):
        """Test minimum variance optimization."""
        result = optimizer.optimize(method=OptimizationMethod.MIN_VARIANCE)
        
        assert isinstance(result, OptimizationResult)
        assert abs(sum(result.weights) - 1.0) < 1e-6
        assert result.volatility > 0
        assert result.volatility < 0.5  # Reasonable volatility
    
    def test_risk_parity_optimization(self, optimizer):
        """Test risk parity optimization."""
        result = optimizer.optimize(method=OptimizationMethod.RISK_PARITY)
        
        assert isinstance(result, OptimizationResult)
        assert 'risk_contributions' in result.metrics
        
        # Risk contributions should be approximately equal
        if result.metrics.get('risk_contributions'):
            rc = np.array(result.metrics['risk_contributions'])
            rc_std = np.std(rc)
            rc_mean = np.mean(rc)
            assert rc_std / rc_mean < 0.5  # Low coefficient of variation
    
    def test_mean_variance_with_target(self, optimizer):
        """Test mean-variance with target return."""
        target = optimizer.mu.mean()
        constraints = OptimizationConstraints(target_return=target)
        
        result = optimizer.optimize(
            method=OptimizationMethod.MEAN_VARIANCE,
            constraints=constraints
        )
        
        assert abs(result.expected_return - target) < 0.01
    
    def test_constraints_allow_short(self, optimizer):
        """Test optimization with short sales allowed."""
        constraints = OptimizationConstraints(allow_short=True)
        
        result = optimizer.optimize(
            method=OptimizationMethod.MAX_SHARPE,
            constraints=constraints
        )
        
        assert abs(sum(result.weights) - 1.0) < 1e-6
        # May have negative weights
    
    def test_constraints_max_position(self, optimizer):
        """Test maximum position size constraint."""
        constraints = OptimizationConstraints(
            allow_short=False,
            max_position_size=0.30
        )
        
        result = optimizer.optimize(
            method=OptimizationMethod.MAX_SHARPE,
            constraints=constraints
        )
        
        assert all(w <= 0.30 + 1e-6 for w in result.weights)
    
    def test_efficient_frontier(self, optimizer):
        """Test efficient frontier generation."""
        frontier = optimizer.efficient_frontier(n_points=20)
        
        assert len(frontier) > 0
        assert 'return' in frontier.columns
        assert 'volatility' in frontier.columns
        
        # Frontier should be convex
        returns = frontier['return'].values
        vols = frontier['volatility'].values
        
        # Generally increasing returns with increasing volatility
        for i in range(1, len(returns)):
            assert returns[i] >= returns[i-1] - 0.01  # Allow small numerical errors
    
    def test_sharpe_ratio_calculation(self, optimizer):
        """Test Sharpe ratio calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        sharpe = optimizer._sharpe_ratio(weights)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_portfolio_volatility_calculation(self, optimizer):
        """Test portfolio volatility calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        vol = optimizer._portfolio_volatility(weights)
        
        assert vol > 0
        assert vol < 1.0  # Reasonable volatility
    
    def test_optimization_result_serialization(self, optimizer):
        """Test result serialization to dict."""
        result = optimizer.optimize(method=OptimizationMethod.MAX_SHARPE)
        data = result.to_dict()
        
        assert 'weights' in data
        assert 'expected_return' in data
        assert 'volatility' in data
        assert 'sharpe_ratio' in data
        assert isinstance(data['weights'], dict)
    
    def test_insufficient_assets_raises_error(self):
        """Test that single asset raises error."""
        returns = pd.DataFrame({'AAPL': [0.01, -0.01, 0.02]})
        
        with pytest.raises(ValueError):
            PortfolioOptimizer(returns)


class TestOptimizationConstraints:
    """Test suite for OptimizationConstraints."""
    
    def test_default_constraints(self):
        """Test default constraint values."""
        c = OptimizationConstraints()
        
        assert c.allow_short == False
        assert c.max_position_size == 1.0
        assert c.min_position_size == 0.0
        assert c.target_return is None
    
    def test_constraint_validation(self):
        """Test constraint validation."""
        c = OptimizationConstraints(max_position_size=0.5)
        assert c.validate() == True
        
        c = OptimizationConstraints(max_position_size=1.5)
        assert c.validate() == False
    
    def test_custom_constraints(self):
        """Test custom constraint configuration."""
        c = OptimizationConstraints(
            allow_short=True,
            max_position_size=0.25,
            target_return=0.15
        )
        
        assert c.allow_short == True
        assert c.max_position_size == 0.25
        assert c.target_return == 0.15
