"""
Black-Litterman Model
======================

Bayesian approach to portfolio optimization combining:
- Market equilibrium returns (CAPM-implied)
- Investor views with confidence levels

References:
-----------
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization
- Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model
- Meucci, A. (2005). Risk and Asset Allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


@dataclass
class InvestorView:
    """Single investor view specification."""
    assets: List[str]  # Assets involved (can be relative: ['AAPL', 'MSFT'] means AAPL vs MSFT)
    weights: List[float]  # Weights (e.g., [1, -1] for AAPL - MSFT)
    return_view: float  # Expected return for this view
    confidence: float  # Confidence level (0 to 1, where 1 = certain)


class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization.
    
    Combines market equilibrium with investor views to produce
    posterior expected returns.
    """
    
    def __init__(self,
                 returns: pd.DataFrame,
                 market_weights: Optional[np.ndarray] = None,
                 risk_aversion: float = 2.5,
                 tau: float = 0.05):
        """
        Initialize Black-Litterman model.
        
        Args:
            returns: Historical returns DataFrame (T x N)
            market_weights: Market capitalization weights (default: equal weight)
            risk_aversion: Risk aversion parameter (lambda)
            tau: Uncertainty scaling factor (typically small, ~0.025-0.05)
        """
        self.returns = returns.dropna()
        self.n_assets = len(returns.columns)
        
        if market_weights is None:
            self.market_weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.market_weights = np.array(market_weights)
        
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # Calculate covariance matrix
        self.cov_matrix = self.returns.cov().values
        self.asset_names = list(returns.columns)
        
        # Prior (equilibrium) returns
        self.pi = None  # Implied equilibrium returns
        self.views = []
        
    def calculate_implied_returns(self) -> np.ndarray:
        """
        Calculate implied equilibrium returns (reverse optimization).
        
        Formula: Π = λΣw_mkt
        where λ = risk aversion, Σ = covariance, w_mkt = market weights
        
        Returns:
            Implied equilibrium returns
        """
        self.pi = self.risk_aversion * self.cov_matrix @ self.market_weights
        return self.pi
    
    def add_view(self, view: InvestorView):
        """
        Add an investor view.
        
        Example:
            # Absolute view: AAPL will return 10%
            view = InvestorView(['AAPL'], [1], 0.10, 0.6)
            
            # Relative view: AAPL will outperform MSFT by 5%
            view = InvestorView(['AAPL', 'MSFT'], [1, -1], 0.05, 0.7)
        """
        self.views.append(view)
    
    def add_absolute_view(self, asset: str, return_view: float, confidence: float):
        """Convenience method for absolute views."""
        self.add_view(InvestorView([asset], [1], return_view, confidence))
    
    def add_relative_view(self, 
                         outperformers: Dict[str, float],
                         underperformers: Dict[str, float],
                         spread: float,
                         confidence: float):
        """
        Convenience method for relative views.
        
        Args:
            outperformers: Dict of asset -> weight (e.g., {'AAPL': 1})
            underperformers: Dict of asset -> weight (e.g., {'MSFT': 1})
            spread: Expected outperformance amount
            confidence: Confidence level
        """
        assets = list(outperformers.keys()) + list(underperformers.keys())
        weights = list(outperformers.values()) + [-w for w in underperformers.values()]
        self.add_view(InvestorView(assets, weights, spread, confidence))
    
    def _build_view_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build P, Q, and Ω matrices from views.
        
        P: View matrix (k x n) - maps views to assets
        Q: View return vector (k)
        Ω: View uncertainty matrix (k x k, diagonal)
        
        Returns:
            Tuple of (P, Q, Omega)
        """
        k = len(self.views)
        n = self.n_assets
        
        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))
        
        for i, view in enumerate(self.views):
            Q[i] = view.return_view
            
            # Map view weights to asset indices
            for asset, weight in zip(view.assets, view.weights):
                if asset in self.asset_names:
                    j = self.asset_names.index(asset)
                    P[i, j] = weight
            
            # View uncertainty (inverse of confidence)
            # Using Idzorek's method: ω = (1-c)/c * PΣP'
            view_var = P[i] @ self.cov_matrix @ P[i]
            Omega[i, i] = (1 - view.confidence) / view.confidence * view_var
        
        return P, Q, Omega
    
    def calculate_posterior_returns(self) -> np.ndarray:
        """
        Calculate posterior expected returns.
        
        Formula:
        E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 × [(τΣ)^-1Π + P'Ω^-1Q]
        
        Returns:
            Posterior expected returns
        """
        if self.pi is None:
            self.calculate_implied_returns()
        
        if len(self.views) == 0:
            logger.warning("No views specified, returning prior returns")
            return self.pi
        
        P, Q, Omega = self._build_view_matrix()
        
        # Prior precision
        tau_sigma = self.tau * self.cov_matrix
        tau_sigma_inv = linalg.inv(tau_sigma)
        
        # View precision
        omega_inv = linalg.inv(Omega)
        
        # Posterior precision
        posterior_prec = tau_sigma_inv + P.T @ omega_inv @ P
        
        # Posterior covariance of mean
        posterior_cov_mean = linalg.inv(posterior_prec)
        
        # Posterior expected returns
        posterior_returns = posterior_cov_mean @ (tau_sigma_inv @ self.pi + P.T @ omega_inv @ Q)
        
        return posterior_returns
    
    def calculate_posterior_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate full posterior distribution.
        
        Returns:
            Tuple of (posterior_mean, posterior_covariance)
        """
        posterior_mean = self.calculate_posterior_returns()
        
        # Posterior covariance of returns (not mean)
        # Σ_post = Σ + [(τΣ)^-1 + P'Ω^-1P]^-1
        if len(self.views) > 0:
            P, Q, Omega = self._build_view_matrix()
            tau_sigma = self.tau * self.cov_matrix
            tau_sigma_inv = linalg.inv(tau_sigma)
            omega_inv = linalg.inv(Omega)
            posterior_cov_mean = linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
            posterior_cov = self.cov_matrix + posterior_cov_mean
        else:
            posterior_cov = self.cov_matrix
        
        return posterior_mean, posterior_cov
    
    def optimize(self, 
                constraints: Optional[Dict] = None,
                allow_short: bool = False) -> Dict:
        """
        Optimize portfolio using Black-Litterman returns.
        
        Args:
            constraints: Optional constraints dict
            allow_short: Whether to allow short positions
            
        Returns:
            Dictionary with optimization results
        """
        from scipy.optimize import minimize
        
        # Get posterior returns and covariance
        mu, Sigma = self.calculate_posterior_distribution()
        
        # Objective: Maximize Sharpe ratio
        def negative_sharpe(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ Sigma @ w)
            return -(port_return - 0.02) / port_vol if port_vol > 0 else 0
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if not allow_short:
            bounds = [(0, 1) for _ in range(self.n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(self.n_assets)]
        
        # Initial guess
        x0 = self.market_weights
        
        # Optimize
        result = minimize(negative_sharpe, x0, method='SLSQP', 
                         bounds=bounds, constraints=cons)
        
        if result.success:
            optimal_weights = result.x
            port_return = optimal_weights @ mu
            port_vol = np.sqrt(optimal_weights @ Sigma @ optimal_weights)
            sharpe = (port_return - 0.02) / port_vol
            
            return {
                'success': True,
                'weights': dict(zip(self.asset_names, optimal_weights)),
                'expected_return': port_return,
                'volatility': port_vol,
                'sharpe_ratio': sharpe,
                'posterior_returns': dict(zip(self.asset_names, mu)),
                'prior_returns': dict(zip(self.asset_names, self.pi))
            }
        else:
            return {'success': False, 'error': result.message}
    
    def get_omega_confidence_matrix(self, confidence_levels: np.ndarray) -> np.ndarray:
        """
        Calculate Omega matrix using Idzorek's confidence method.
        
        Alternative to specifying Omega directly.
        
        Args:
            confidence_levels: Array of confidence levels (0-1) for each view
            
        Returns:
            Omega matrix
        """
        P, Q, _ = self._build_view_matrix()
        Omega = np.zeros((len(self.views), len(self.views)))
        
        for i, conf in enumerate(confidence_levels):
            view_var = P[i] @ self.cov_matrix @ P[i]
            Omega[i, i] = (1 - conf) / conf * view_var
        
        return Omega


class BlackLittermanViewsGUI:
    """
    Helper class to generate common Black-Litterman views.
    """
    
    @staticmethod
    def bullish_market(assets: List[str], 
                      excess_return: float = 0.03,
                      confidence: float = 0.5) -> List[InvestorView]:
        """Generate views for bullish market outlook."""
        views = []
        for asset in assets:
            views.append(InvestorView([asset], [1], excess_return, confidence))
        return views
    
    @staticmethod
    def sector_rotation(from_sectors: Dict[str, float],
                       to_sectors: Dict[str, float],
                       spread: float = 0.05,
                       confidence: float = 0.4) -> InvestorView:
        """Generate sector rotation view."""
        assets = list(to_sectors.keys()) + list(from_sectors.keys())
        weights = list(to_sectors.values()) + [-w for w in from_sectors.values()]
        return InvestorView(assets, weights, spread, confidence)
    
    @staticmethod
    def quality_factor(quality_assets: List[str],
                      junk_assets: List[str],
                      spread: float = 0.04,
                      confidence: float = 0.45) -> InvestorView:
        """Generate quality vs junk view."""
        n_quality = len(quality_assets)
        n_junk = len(junk_assets)
        
        assets = quality_assets + junk_assets
        weights = [1/n_quality] * n_quality + [-1/n_junk] * n_junk
        
        return InvestorView(assets, weights, spread, confidence)


def create_tactical_views(economic_regime: str,
                         sector_preferences: Dict[str, float]) -> List[InvestorView]:
    """
    Create Black-Litterman views based on economic regime.
    
    Args:
        economic_regime: 'expansion', 'contraction', 'recovery', 'peak'
        sector_preferences: Dict of sector -> preference weight
        
    Returns:
        List of investor views
    """
    views = []
    
    regime_views = {
        'expansion': {
            'favored': ['Technology', 'Consumer Discretionary', 'Industrials'],
            'avoided': ['Utilities', 'Consumer Staples'],
            'confidence': 0.5
        },
        'contraction': {
            'favored': ['Utilities', 'Consumer Staples', 'Healthcare'],
            'avoided': ['Technology', 'Consumer Discretionary'],
            'confidence': 0.6
        },
        'recovery': {
            'favored': ['Financials', 'Industrials', 'Materials'],
            'avoided': ['Utilities'],
            'confidence': 0.45
        },
        'peak': {
            'favored': ['Healthcare', 'Energy', 'Materials'],
            'avoided': ['Technology', 'Financials'],
            'confidence': 0.4
        }
    }
    
    regime = regime_views.get(economic_regime, regime_views['expansion'])
    
    # Add relative views between favored and avoided sectors
    for favored in regime['favored']:
        for avoided in regime['avoided']:
            if favored in sector_preferences and avoided in sector_preferences:
                views.append(InvestorView(
                    assets=[favored, avoided],
                    weights=[1, -1],
                    return_view=0.03,  # 3% outperformance
                    confidence=regime['confidence']
                ))
    
    return views
