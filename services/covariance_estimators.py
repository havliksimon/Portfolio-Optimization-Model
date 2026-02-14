"""
Advanced Covariance Matrix Estimators
======================================

Improved covariance estimation methods:
- Ledoit-Wolf shrinkage
- Oracle Approximating Shrinkage (OAS)
- Graphical Lasso
- Factor models for covariance

References:
-----------
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices
- Ledoit, O. & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix
- Chen, Y. et al. (2010). Shrinkage algorithms for MMSE covariance estimation
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


class LedoitWolfShrinkage:
    """
    Ledoit-Wolf shrinkage estimator for covariance matrices.
    
    Combines sample covariance with a structured target to improve
    conditioning and reduce estimation error.
    """
    
    TARGETS = ['constant_correlation', 'single_factor', 'identity', 'diagonal']
    
    def __init__(self, target: Literal['constant_correlation', 'single_factor', 'identity', 'diagonal'] = 'constant_correlation'):
        """
        Initialize shrinkage estimator.
        
        Args:
            target: Shrinkage target type
                - 'constant_correlation': All pairs have same correlation
                - 'single_factor': Market model structure
                - 'identity': Uncorrelated with unit variance
                - 'diagonal': Uncorrelated with sample variances
        """
        self.target = target
        self.shrinkage_factor = None
        self.sample_cov = None
        self.target_cov = None
        self.shrunk_cov = None
        
    def fit(self, returns: pd.DataFrame) -> 'LedoitWolfShrinkage':
        """
        Fit shrinkage estimator.
        
        Args:
            returns: T x N DataFrame of returns
            
        Returns:
            self
        """
        X = returns.dropna().values
        T, N = X.shape
        
        # Sample covariance
        X_centered = X - X.mean(axis=0)
        self.sample_cov = (X_centered.T @ X_centered) / T
        
        # Build target
        if self.target == 'constant_correlation':
            self.target_cov = self._constant_correlation_target(self.sample_cov)
        elif self.target == 'single_factor':
            self.target_cov = self._single_factor_target(X)
        elif self.target == 'identity':
            self.target_cov = np.eye(N)
        elif self.target == 'diagonal':
            self.target_cov = np.diag(np.diag(self.sample_cov))
        
        # Calculate optimal shrinkage intensity
        self.shrinkage_factor = self._optimal_shrinkage(X, self.sample_cov, self.target_cov)
        
        # Compute shrunk covariance
        self.shrunk_cov = (self.shrinkage_factor * self.target_cov + 
                          (1 - self.shrinkage_factor) * self.sample_cov)
        
        return self
    
    def _constant_correlation_target(self, S: np.ndarray) -> np.ndarray:
        """Build constant correlation target."""
        n = S.shape[0]
        
        # Average correlation
        corr = self._cov_to_corr(S)
        avg_corr = (np.sum(corr) - n) / (n * (n - 1))
        
        # Target: sample variances, constant correlation
        target = np.zeros_like(S)
        variances = np.diag(S)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    target[i, j] = variances[i]
                else:
                    target[i, j] = avg_corr * np.sqrt(variances[i] * variances[j])
        
        return target
    
    def _single_factor_target(self, X: np.ndarray) -> np.ndarray:
        """Build single-factor (market model) target."""
        # Use equal-weighted market as factor
        market = X.mean(axis=1)
        
        # Regress each asset on market
        betas = []
        residuals = []
        
        for i in range(X.shape[1]):
            # Simple regression
            beta = np.cov(X[:, i], market)[0, 1] / np.var(market)
            betas.append(beta)
            resid = X[:, i] - beta * market
            residuals.append(resid)
        
        betas = np.array(betas)
        residuals = np.array(residuals)
        
        # Factor covariance
        market_var = np.var(market)
        specific_var = np.var(residuals, axis=1)
        
        target = np.outer(betas, betas) * market_var + np.diag(specific_var)
        
        return target
    
    def _optimal_shrinkage(self, X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
        """
        Calculate optimal shrinkage intensity.
        
        Using Ledoit-Wolf formula.
        """
        T, N = X.shape
        X_centered = X - X.mean(axis=0)
        
        # Frobenius norm squared of difference
        delta = np.sum((S - F) ** 2)
        
        # Calculate pi (asymptotic variance term)
        pi = 0
        for t in range(T):
            Yt = np.outer(X_centered[t], X_centered[t])
            pi += np.sum((Yt - S) ** 2)
        pi /= T
        
        # Calculate rho (bias term)
        gamma = np.sum((F - S) ** 2)
        rho = pi - gamma
        
        # Optimal shrinkage
        kappa = max(0, (pi - rho) / gamma) if gamma > 0 else 0
        shrinkage = min(1, kappa / T)
        
        return shrinkage
    
    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance to correlation matrix."""
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)
    
    def covariance(self) -> pd.DataFrame:
        """Return shrunk covariance as DataFrame."""
        if self.shrunk_cov is None:
            raise ValueError("Must call fit() first")
        return pd.DataFrame(self.shrunk_cov, 
                          index=self.sample_cov.shape,
                          columns=self.sample_cov.shape)


class OracleApproximatingShrinkage:
    """
    Oracle Approximating Shrinkage (OAS) estimator.
    
    Alternative to Ledoit-Wolf with better finite-sample properties.
    Assumes Gaussian data.
    """
    
    def __init__(self):
        self.shrinkage_factor = None
        self.shrunk_cov = None
        
    def fit(self, returns: pd.DataFrame) -> 'OracleApproximatingShrinkage':
        """Fit OAS estimator."""
        X = returns.dropna().values
        T, N = X.shape
        
        # Sample covariance (unbiased)
        X_centered = X - X.mean(axis=0)
        S = (X_centered.T @ X_centered) / (T - 1)
        
        # Trace of S
        tr_S = np.trace(S)
        tr_S2 = np.trace(S @ S)
        
        # OAS shrinkage formula
        num = (1 - 2/N) * tr_S2 + tr_S**2
        den = (T + 1 - 2/N) * (tr_S2 - tr_S**2 / N)
        
        if den > 0:
            self.shrinkage_factor = min(1, num / den)
        else:
            self.shrinkage_factor = 1
        
        # Target: scaled identity
        target = (tr_S / N) * np.eye(N)
        
        self.shrunk_cov = (self.shrinkage_factor * target + 
                          (1 - self.shrinkage_factor) * S)
        
        return self


class FactorCovarianceEstimator:
    """
    Covariance estimation using factor models.
    
    Reduces dimensionality by assuming returns follow a factor structure.
    """
    
    def __init__(self, n_factors: int = 5, method: str = 'pca'):
        """
        Initialize factor covariance estimator.
        
        Args:
            n_factors: Number of factors
            method: 'pca' or 'macro' (for predefined factors)
        """
        self.n_factors = n_factors
        self.method = method
        self.factor_loadings = None
        self.factor_cov = None
        self.specific_cov = None
        
    def fit(self, returns: pd.DataFrame, factors: Optional[pd.DataFrame] = None) -> 'FactorCovarianceEstimator':
        """
        Fit factor model.
        
        Args:
            returns: Asset returns
            factors: Predefined factors (if method='macro')
            
        Returns:
            self
        """
        if self.method == 'pca':
            self._fit_pca(returns)
        elif self.method == 'macro' and factors is not None:
            self._fit_macro(returns, factors)
        else:
            raise ValueError("Must provide factors for macro method")
        return self
    
    def _fit_pca(self, returns: pd.DataFrame):
        """Fit using PCA."""
        from sklearn.decomposition import PCA
        
        X = returns.dropna().values
        
        # PCA
        pca = PCA(n_components=self.n_factors)
        factor_returns = pca.fit_transform(X)
        
        # Factor loadings
        self.factor_loadings = pca.components_.T  # N x K
        self.factor_cov = np.cov(factor_returns.T)  # K x K
        
        # Specific returns (residuals)
        explained = factor_returns @ self.factor_loadings.T
        residuals = X - explained
        self.specific_cov = np.diag(np.var(residuals, axis=0))
        
    def _fit_macro(self, returns: pd.DataFrame, factors: pd.DataFrame):
        """Fit using predefined macro factors."""
        # Align dates
        aligned = pd.concat([returns, factors], axis=1).dropna()
        
        R = aligned.iloc[:, :len(returns.columns)].values
        F = aligned.iloc[:, len(returns.columns):].values
        
        # Regression for loadings
        F_augmented = np.column_stack([np.ones(len(F)), F])
        
        loadings = []
        residuals = []
        
        for i in range(R.shape[1]):
            beta = linalg.lstsq(F_augmented, R[:, i], rcond=None)[0]
            loadings.append(beta[1:])  # Exclude intercept
            resid = R[:, i] - F_augmented @ beta
            residuals.append(resid)
        
        self.factor_loadings = np.array(loadings)  # N x K
        self.factor_cov = np.cov(F.T)  # K x K
        self.specific_cov = np.diag(np.var(np.array(residuals), axis=1))
    
    def covariance(self) -> np.ndarray:
        """Return full covariance matrix."""
        return (self.factor_loadings @ self.factor_cov @ self.factor_loadings.T + 
                self.specific_cov)
    
    def correlation(self) -> np.ndarray:
        """Return correlation matrix."""
        cov = self.covariance()
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)


class CovarianceMixture:
    """
    Robust covariance using regime-dependent mixtures.
    
    Models covariance as mixture of different market regimes.
    """
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.regime_covs = []
        self.regime_probs = []
        
    def fit(self, returns: pd.DataFrame, regime_labels: Optional[np.ndarray] = None):
        """
        Fit mixture covariance.
        
        Args:
            returns: Returns DataFrame
            regime_labels: Optional predefined regime labels
        """
        if regime_labels is None:
            # Use simple volatility-based regime detection
            vol = returns.std(axis=1)
            high_vol = vol > vol.median()
            regime_labels = high_vol.astype(int).values
        
        # Calculate covariance for each regime
        for r in range(self.n_regimes):
            mask = regime_labels == r
            if mask.sum() > 10:  # Minimum observations
                regime_returns = returns[mask]
                self.regime_covs.append(regime_returns.cov().values)
                self.regime_probs.append(mask.mean())
            else:
                # Fallback to overall covariance
                self.regime_covs.append(returns.cov().values)
                self.regime_probs.append(1.0 / self.n_regimes)
        
        # Normalize probabilities
        total_prob = sum(self.regime_probs)
        self.regime_probs = [p / total_prob for p in self.regime_probs]
    
    def covariance(self) -> np.ndarray:
        """Return unconditional covariance (mixture)."""
        result = np.zeros_like(self.regime_covs[0])
        for cov, prob in zip(self.regime_covs, self.regime_probs):
            result += prob * cov
        return result
    
    def regime_covariance(self, regime: int) -> np.ndarray:
        """Return covariance for specific regime."""
        return self.regime_covs[regime]


def compare_estimators(returns: pd.DataFrame, test_size: int = 63) -> pd.DataFrame:
    """
    Compare different covariance estimators using out-of-sample likelihood.
    
    Args:
        returns: Returns DataFrame
        test_size: Number of observations to use for testing
        
    Returns:
        DataFrame with comparison metrics
    """
    train = returns.iloc[:-test_size]
    test = returns.iloc[-test_size:]
    
    results = []
    
    # Sample covariance
    sample_cov = train.cov().values
    results.append({
        'estimator': 'Sample',
        'log_likelihood': _gaussian_log_likelihood(test, sample_cov),
        'condition_number': np.linalg.cond(sample_cov)
    })
    
    # Ledoit-Wolf
    lw = LedoitWolfShrinkage('constant_correlation').fit(train)
    results.append({
        'estimator': 'Ledoit-Wolf (CC)',
        'log_likelihood': _gaussian_log_likelihood(test, lw.shrunk_cov),
        'condition_number': np.linalg.cond(lw.shrunk_cov),
        'shrinkage': lw.shrinkage_factor
    })
    
    # OAS
    oas = OracleApproximatingShrinkage().fit(train)
    results.append({
        'estimator': 'OAS',
        'log_likelihood': _gaussian_log_likelihood(test, oas.shrunk_cov),
        'condition_number': np.linalg.cond(oas.shrunk_cov),
        'shrinkage': oas.shrinkage_factor
    })
    
    # Factor model (use min of 5 or n_assets-1)
    n_factors = min(5, len(train.columns) - 1)
    if n_factors > 0:
        factor = FactorCovarianceEstimator(n_factors=n_factors).fit(train)
        results.append({
            'estimator': 'Factor (PCA)',
            'log_likelihood': _gaussian_log_likelihood(test, factor.covariance()),
            'condition_number': np.linalg.cond(factor.covariance())
        })
    
    return pd.DataFrame(results)


def _gaussian_log_likelihood(returns: pd.DataFrame, cov: np.ndarray) -> float:
    """Calculate Gaussian log-likelihood."""
    X = returns.values
    n = len(X)
    k = X.shape[1]
    
    # Log determinant
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    
    # Mahalanobis distance
    inv_cov = linalg.inv(cov)
    mean = X.mean(axis=0)
    diff = X - mean
    mahal = np.sum(diff @ inv_cov * diff)
    
    return -0.5 * (n * k * np.log(2 * np.pi) + n * logdet + mahal)
