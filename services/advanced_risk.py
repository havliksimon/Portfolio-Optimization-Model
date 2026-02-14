"""
Advanced Portfolio Risk Analytics
==================================

Extended risk metrics and quantitative finance features:
- Modified VaR (Cornish-Fisher expansion)
- Kelly Criterion and Half-Kelly
- Ulcer Index and related metrics
- Incremental and Component VaR
- Drawdown-at-Risk metrics

References:
-----------
- Favre, L. & Galeano, J.A. (2002). Mean-Modified Value-at-Risk
- Kelly, J.L. (1956). A New Interpretation of Information Rate
- Martin, P. & McCann, B. (1989). The Investor's Guide to Fidelity Funds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdvancedRiskMetrics:
    """Extended risk metrics beyond standard VaR/CVaR."""
    
    # Modified VaR (Cornish-Fisher expansion)
    mvar_95: float
    mvar_99: float
    
    # Kelly Criterion
    kelly_fraction: float
    half_kelly_fraction: float
    optimal_leverage: float
    growth_rate: float
    
    # Ulcer Index metrics
    ulcer_index: float
    pain_ratio: float
    
    # Drawdown metrics
    avg_drawdown: float
    drawdown_std: float
    calmar_ratio_3y: float
    
    # Incremental VaR
    ivar_contributions: Dict[str, float]
    
    # Component VaR
    cvar_contributions: Dict[str, float]
    
    # Tail risk
    tail_ratio_95_5: float
    kappa_3: float  # Third-order Lower Partial Moment
    

class AdvancedRiskCalculator:
    """Calculate advanced risk metrics for portfolios."""
    
    def __init__(self, returns: pd.DataFrame, weights: Optional[np.ndarray] = None):
        """
        Initialize calculator.
        
        Args:
            returns: DataFrame of asset returns (columns = assets)
            weights: Portfolio weights (if None, assumes equal weight)
        """
        self.returns = returns.dropna()
        self.n_assets = len(returns.columns)
        
        if weights is None:
            self.weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.weights = np.array(weights)
            
        # Calculate portfolio returns
        self.port_returns = self.returns @ self.weights
        
    def modified_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Modified VaR using Cornish-Fisher expansion.
        
        Adjusts for skewness and kurtosis in the return distribution.
        
        Formula:
        MVaR = μ + σ × z_cf
        where z_cf = z + (z² - 1)S/6 + (z³ - 3z)(K-3)/24 - (2z³ - 5z)S²/36
        
        Args:
            confidence: Confidence level (default 0.95)
            
        Returns:
            Modified VaR value
        """
        z = stats.norm.ppf(1 - confidence)
        mu = self.port_returns.mean()
        sigma = self.port_returns.std()
        S = self.port_returns.skew()
        K = self.port_returns.kurtosis()
        
        # Cornish-Fisher expansion
        z_cf = (z + 
                (z**2 - 1) * S / 6 +
                (z**3 - 3*z) * (K - 3) / 24 -
                (2*z**3 - 5*z) * S**2 / 36)
        
        return mu + sigma * z_cf
    
    def kelly_criterion(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate Kelly Criterion for optimal position sizing.
        
        The Kelly fraction represents the optimal proportion of capital
        to allocate to maximize expected log wealth.
        
        Formula:
        f* = (μ - r) / σ²
        
        For multiple assets:
        f* = Σ⁻¹(μ - r)
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary with Kelly metrics
        """
        # Annualized returns and covariance
        mu = self.returns.mean() * 252
        sigma = self.returns.cov() * 252
        excess_returns = mu - risk_free_rate
        
        try:
            # Kelly fraction for full portfolio
            kelly = np.linalg.solve(sigma, excess_returns)
            
            # Portfolio-level Kelly (for the weighted portfolio)
            port_mean = self.port_returns.mean() * 252
            port_var = self.port_returns.var() * 252
            port_kelly = (port_mean - risk_free_rate) / port_var if port_var > 0 else 0
            
            # Growth rate (expected log return)
            growth = risk_free_rate + port_kelly * (port_mean - risk_free_rate) - \
                     0.5 * port_kelly**2 * port_var
            
            return {
                'kelly_fraction': float(port_kelly),
                'half_kelly_fraction': float(port_kelly / 2),
                'optimal_leverage': float(port_kelly),
                'growth_rate': float(growth),
                'kelly_by_asset': kelly.tolist()
            }
        except np.linalg.LinAlgError:
            return {
                'kelly_fraction': 0.0,
                'half_kelly_fraction': 0.0,
                'optimal_leverage': 0.0,
                'growth_rate': 0.0,
                'kelly_by_asset': [0.0] * self.n_assets
            }
    
    def ulcer_index(self, window: int = 252) -> Dict[str, float]:
        """
        Calculate Ulcer Index and related metrics.
        
        The Ulcer Index measures downside risk by focusing on drawdowns
        rather than volatility.
        
        Formula:
        UI = sqrt(mean(R²)) where R = (Peak - Price) / Peak × 100
        
        Args:
            window: Rolling window for calculation
            
        Returns:
            Dictionary with Ulcer metrics
        """
        # Calculate drawdowns
        cum_returns = (1 + self.port_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        
        # Ulcer Index (square root of mean squared drawdown)
        ui = np.sqrt(np.mean(drawdown**2))
        
        # Alternative: Martin Ratio (return / UI)
        ann_return = self.port_returns.mean() * 252
        martin_ratio = ann_return / ui if ui > 0 else 0
        
        # Pain Index (average drawdown)
        pain_index = -drawdown.mean()
        
        # Pain Ratio (return / Pain Index)
        pain_ratio = ann_return / pain_index if pain_index > 0 else 0
        
        return {
            'ulcer_index': float(ui),
            'martin_ratio': float(martin_ratio),
            'pain_index': float(pain_index),
            'pain_ratio': float(pain_ratio),
            'max_drawdown': float(drawdown.min())
        }
    
    def incremental_var(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate Incremental VaR for each position.
        
        IVaR measures the change in portfolio VaR when removing a position.
        
        Args:
            confidence: Confidence level
            
        Returns:
            Dictionary mapping asset names to IVaR
        """
        from scipy.optimize import minimize_scalar
        
        # Current portfolio VaR
        current_var = self._calculate_var(self.port_returns, confidence)
        
        ivar = {}
        for i, col in enumerate(self.returns.columns):
            # Create new weights with position i removed (redistribute)
            new_weights = self.weights.copy()
            removed_weight = new_weights[i]
            new_weights[i] = 0
            
            if removed_weight < 1.0:
                # Redistribute to remaining positions
                mask = new_weights > 0
                if mask.any():
                    new_weights[mask] += removed_weight / mask.sum()
            
            # New portfolio returns
            new_port_returns = self.returns @ new_weights
            new_var = self._calculate_var(new_port_returns, confidence)
            
            # IVaR is the difference
            ivar[col] = float(new_var - current_var)
        
        return ivar
    
    def component_var(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate Component VaR for each position.
        
        CVaR is the contribution of each position to total portfolio VaR.
        
        Formula:
        CVaR_i = w_i × β_i × Portfolio_VaR
        where β_i = Cov(r_i, r_p) / Var(r_p)
        
        Args:
            confidence: Confidence level
            
        Returns:
            Dictionary mapping asset names to CVaR
        """
        # Portfolio VaR
        port_var = self._calculate_var(self.port_returns, confidence)
        
        # Marginal VaR (beta to portfolio)
        cov_with_port = self.returns.cov().dot(self.weights)
        port_variance = self.port_returns.var()
        betas = cov_with_port / port_variance if port_variance > 0 else 0
        
        # Component VaR
        cvar = {}
        for i, col in enumerate(self.returns.columns):
            cvar[col] = float(self.weights[i] * betas.iloc[i] * port_var)
        
        return cvar
    
    def drawdown_at_risk(self, confidence: float = 0.95, window: int = 252) -> Dict[str, float]:
        """
        Calculate Drawdown-at-Risk metrics.
        
        Similar to VaR but for drawdowns instead of returns.
        
        Args:
            confidence: Confidence level
            window: Rolling window
            
        Returns:
            Dictionary with DaR metrics
        """
        # Calculate rolling drawdowns
        cum_returns = (1 + self.port_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        
        # Drawdown at Risk
        dar = np.percentile(drawdown.dropna(), (1 - confidence) * 100)
        
        # Conditional Drawdown at Risk (CDaR)
        cdar = drawdown[drawdown <= dar].mean() if len(drawdown[drawdown <= dar]) > 0 else dar
        
        # Average drawdown
        avg_dd = drawdown.mean()
        std_dd = drawdown.std()
        
        return {
            'dar_95': float(dar),
            'cdar_95': float(cdar),
            'avg_drawdown': float(avg_dd),
            'drawdown_std': float(std_dd),
            'ulcer_index': float(np.sqrt(np.mean(drawdown**2)))
        }
    
    def tail_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate various tail risk metrics.
        
        Returns:
            Dictionary with tail risk measures
        """
        returns = self.port_returns.dropna()
        
        # Tail ratio (95th percentile / 5th percentile)
        tail_95 = np.percentile(returns, 95)
        tail_5 = np.percentile(returns, 5)
        tail_ratio = abs(tail_95 / tail_5) if tail_5 != 0 else 0
        
        # Kappa 3 (third-order lower partial moment)
        target = 0
        lpm_3 = np.mean(np.maximum(target - returns, 0)**3)
        excess_return = returns.mean() * 252 - target
        kappa_3 = excess_return / (lpm_3**(1/3)) if lpm_3 > 0 else 0
        
        # Burke ratio (return / sqrt(sum of squared drawdowns))
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = ((cum_returns - rolling_max) / rolling_max).dropna()
        top_5_dd = drawdowns.nsmallest(5)**2
        burke_denom = np.sqrt(top_5_dd.sum()) if len(top_5_dd) > 0 else 1
        burke_ratio = (returns.mean() * 252) / burke_denom if burke_denom > 0 else 0
        
        # Sterling ratio (return / average of 5 largest drawdowns)
        sterling_denom = -drawdowns.nsmallest(5).mean() if len(drawdowns) > 0 else 1
        sterling_ratio = (returns.mean() * 252) / sterling_denom if sterling_denom > 0 else 0
        
        return {
            'tail_ratio_95_5': float(tail_ratio),
            'kappa_3': float(kappa_3),
            'burke_ratio': float(burke_ratio),
            'sterling_ratio': float(sterling_ratio),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'jarque_bera_stat': float(stats.jarque_bera(returns)[0]),
            'jarque_bera_pvalue': float(stats.jarque_bera(returns)[1])
        }
    
    def calculate_all(self) -> AdvancedRiskMetrics:
        """Calculate all advanced risk metrics."""
        mvar_95 = self.modified_var(0.95)
        mvar_99 = self.modified_var(0.99)
        
        kelly = self.kelly_criterion()
        ulcer = self.ulcer_index()
        
        ivar = self.incremental_var()
        cvar = self.component_var()
        
        dar = self.drawdown_at_risk()
        tail = self.tail_risk_metrics()
        
        return AdvancedRiskMetrics(
            mvar_95=mvar_95,
            mvar_99=mvar_99,
            kelly_fraction=kelly['kelly_fraction'],
            half_kelly_fraction=kelly['half_kelly_fraction'],
            optimal_leverage=kelly['optimal_leverage'],
            growth_rate=kelly['growth_rate'],
            ulcer_index=ulcer['ulcer_index'],
            pain_ratio=ulcer['pain_ratio'],
            avg_drawdown=dar['avg_drawdown'],
            drawdown_std=dar['drawdown_std'],
            calmar_ratio_3y=0.0,  # Would need 3 years of data
            ivar_contributions=ivar,
            cvar_contributions=cvar,
            tail_ratio_95_5=tail['tail_ratio_95_5'],
            kappa_3=tail['kappa_3']
        )
    
    @staticmethod
    def _calculate_var(returns: pd.Series, confidence: float) -> float:
        """Helper to calculate historical VaR."""
        return np.percentile(returns.dropna(), (1 - confidence) * 100)


def calculate_hierarchical_risk_parity(returns: pd.DataFrame, 
                                       method: str = 'single') -> pd.Series:
    """
    Calculate Hierarchical Risk Parity (HRP) portfolio weights.
    
    Based on Lopez de Prado (2016), uses hierarchical clustering
    to allocate risk without inverting the covariance matrix.
    
    Algorithm:
    1. Compute distance matrix from correlation
    2. Perform hierarchical clustering
    3. Quasi-diagonalization (seriation)
    4. Recursive bisection for weight allocation
    
    Args:
        returns: DataFrame of asset returns
        method: Linkage method ('single', 'complete', 'average', 'ward')
        
    Returns:
        Series of optimal weights
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    
    # 1. Compute correlation and distance
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    
    # 2. Hierarchical clustering
    linkage_matrix = linkage(squareform(dist), method=method)
    
    # 3. Quasi-diagonalization
    sorted_idx = leaves_list(linkage_matrix)
    sorted_tickers = [returns.columns[i] for i in sorted_idx]
    
    # 4. Recursive bisection
    weights = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]
    
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            
            # Split cluster in half
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            
            # Calculate cluster variance
            left_var = returns[left].var().mean()
            right_var = returns[right].var().mean()
            
            # Allocate weight inversely proportional to variance
            total_var = left_var + right_var
            if total_var > 0:
                left_weight = 1 - left_var / total_var
                right_weight = 1 - right_var / total_var
            else:
                left_weight = right_weight = 0.5
            
            # Apply weights
            weights[left] *= left_weight
            weights[right] *= right_weight
            
            new_clusters.extend([left, right])
        
        clusters = new_clusters
    
    # Normalize to sum to 1
    weights = weights / weights.sum()
    
    # Return in original order
    return weights[returns.columns]
