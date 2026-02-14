# Advanced Portfolio Analytics Documentation

## Overview

This portfolio optimizer includes cutting-edge quantitative finance techniques across multiple domains:

- **Risk Analytics**: VaR, CVaR, Modified VaR, Tail Risk
- **Portfolio Optimization**: Mean-Variance, Risk Parity, Black-Litterman, Hierarchical Risk Parity
- **Factor Models**: Fama-French, Statistical Factor Analysis
- **Regime Detection**: Hidden Markov Models, Trend Following
- **Machine Learning**: Hierarchical Clustering, PCA, Covariance Estimation

---

## API Endpoints

### Core Endpoints

#### `POST /api/portfolio/comprehensive-analysis`
Full portfolio analysis including:
- Risk metrics (VaR, CVaR, Sharpe, Sortino, etc.)
- Monte Carlo simulation (5,000 paths)
- Distribution fitting (Normal, Student-t, Laplace)
- Statistical tests (Jarque-Bera, Ljung-Box, ARCH)
- PCA for diversification analysis
- Stress testing
- Tail risk analysis (EVT)

**Request:**
```json
{
  "holdings": [
    {"ticker": "AAPL", "weight": 0.3},
    {"ticker": "MSFT", "weight": 0.3},
    {"ticker": "GOOGL", "weight": 0.4}
  ],
  "period": "2y"
}
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "total_return": 0.3738,
    "volatility": 0.2143,
    "sharpe_ratio": 0.57,
    "var_95": -0.0211,
    "cvar_95": -0.0303
  },
  "monte_carlo": {
    "probability_profit": 0.7296,
    "expected_max_drawdown": -0.15,
    "paths": [...]
  },
  "pca": {
    "explained_variance_ratio": [0.6138, 0.1853, ...]
  }
}
```

---

### Advanced Risk Endpoints

#### `POST /api/advanced/risk`
Extended risk metrics including:
- Modified VaR (Cornish-Fisher expansion)
- Kelly Criterion for optimal position sizing
- Ulcer Index and Pain Ratio
- Drawdown-at-Risk (DaR)
- Tail risk metrics (Kappa, Burke ratio)

**Example Response:**
```json
{
  "modified_var": {
    "var_95": -0.0151,
    "var_99": -0.0234
  },
  "kelly_criterion": {
    "kelly_fraction": 3.32,
    "half_kelly_fraction": 1.66,
    "optimal_leverage": 3.32
  },
  "ulcer_metrics": {
    "ulcer_index": 0.045,
    "pain_ratio": 1.23
  }
}
```

---

### Advanced Optimization Endpoints

#### `POST /api/hierarchical-risk-parity`
Machine learning-based allocation (Lopez de Prado, 2016):
- Uses hierarchical clustering instead of covariance inversion
- More robust for correlated assets
- No matrix inversion required

**Example:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "period": "1y"
}
```

**Response:**
```json
{
  "weights": {
    "AAPL": 0.25,
    "MSFT": 0.25,
    "GOOGL": 0.25,
    "AMZN": 0.25
  },
  "diversification_ratio": 4.0
}
```

#### `POST /api/black-litterman`
Bayesian portfolio optimization combining market equilibrium with investor views.

**Example:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "views": [
    {"asset": "AAPL", "expected_return": 0.15, "confidence": 0.6}
  ],
  "period": "2y"
}
```

---

### Factor Analysis Endpoints

#### `POST /api/factor-exposure`
Estimate portfolio exposure to Fama-French factors:
- Market beta
- SMB (Small minus Big)
- HML (High minus Low / Value)
- RMW (Robust minus Weak / Profitability)
- CMA (Conservative minus Aggressive / Investment)
- MOM (Momentum - Carhart)

**Example Response:**
```json
{
  "exposure": {
    "market_beta": 1.05,
    "smb_beta": -0.15,
    "hml_beta": 0.23,
    "alpha": 0.023,
    "alpha_significant": false,
    "r_squared": 0.85
  }
}
```

---

### Market Regime Endpoints

#### `POST /api/regime-detection`
Hidden Markov Model for detecting market regimes:
- Bull vs Bear markets
- High vs Low volatility periods
- Regime characteristics (mean, vol, VaR)

**Example Response:**
```json
{
  "n_regimes": 2,
  "current_regime": "Regime_0",
  "regime_probabilities": {
    "Regime_0": 0.65,
    "Regime_1": 0.35
  },
  "regime_characteristics": [
    {
      "name": "Regime_0",
      "mean": 0.0005,
      "volatility": 0.012,
      "var_95": -0.018
    }
  ]
}
```

---

### Covariance Estimation Endpoints

#### `POST /api/covariance-estimators`
Compare different covariance matrix estimators:
- Sample covariance
- Ledoit-Wolf shrinkage
- Oracle Approximating Shrinkage (OAS)
- Factor models (PCA)

**Example Response:**
```json
{
  "comparison": [
    {
      "estimator": "Sample",
      "log_likelihood": -1250.5,
      "condition_number": 145.2
    },
    {
      "estimator": "Ledoit-Wolf (CC)",
      "log_likelihood": -1180.3,
      "condition_number": 45.1,
      "shrinkage": 0.35
    }
  ]
}
```

---

## Mathematical Background

### Modified VaR (Cornish-Fisher Expansion)

Standard VaR assumes normal distribution. Modified VaR adjusts for skewness and kurtosis:

```
MVaR = μ + σ × z_cf

where z_cf = z + (z² - 1)S/6 + (z³ - 3z)(K-3)/24 - (2z³ - 5z)S²/36

z = normal quantile
S = skewness
K = kurtosis
```

### Kelly Criterion

Optimal fraction to maximize expected log wealth:

```
f* = (μ - r) / σ²

For multiple assets:
f* = Σ⁻¹(μ - r)
```

### Black-Litterman Model

Combines market equilibrium (Π) with investor views (Q):

```
E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹Π + P'Ω⁻¹Q]

where:
Π = λΣw_mkt (implied equilibrium returns)
τ = uncertainty parameter (~0.05)
P = view matrix
Ω = view uncertainty
```

### Hierarchical Risk Parity

Algorithm:
1. Compute distance matrix: d_ij = √(0.5 × (1 - ρ_ij))
2. Perform hierarchical clustering
3. Quasi-diagonalization (seriation)
4. Recursive bisection for weight allocation

### Hidden Markov Model

For regime detection:

```
P(S_t | X) ∝ P(X_t | S_t) × Σ_s P(S_t | S_{t-1}=s) × P(S_{t-1}=s | X_{1:t-1})

where:
S_t = hidden state at time t
X_t = observation (return)
P(X_t | S_t) = emission probability (Gaussian)
P(S_t | S_{t-1}) = transition probability
```

---

## Implementation Details

### Monte Carlo Simulation

```python
# Bootstrap method
for path in range(n_simulations):
    sampled_returns = np.random.choice(historical_returns, n_days)
    price_path = initial_value * (1 + sampled_returns).cumprod()
```

### Extreme Value Theory

For tail risk:
```python
# Hill estimator for tail index
sorted_losses = np.sort(losses)[::-1]  # descending
k = int(0.05 * n)  # top 5% losses
tail_index = np.mean(np.log(sorted_losses[:k] / sorted_losses[k]))
```

### PCA Factor Model

```python
# Extract latent factors
pca = PCA(n_components=5)
factors = pca.fit_transform(returns)

# Reconstruct covariance
Σ = B @ Σ_factors @ B' + Σ_specific
```

---

## Usage Examples

### Python Client

```python
import requests

# Comprehensive analysis
response = requests.post('http://localhost:5000/api/portfolio/comprehensive-analysis', json={
    'holdings': [
        {'ticker': 'AAPL', 'weight': 0.4},
        {'ticker': 'MSFT', 'weight': 0.6}
    ],
    'period': '2y'
})
data = response.json()

print(f"VaR 95%: {data['metrics']['var_95']:.2%}")
print(f"Expected Return: {data['metrics']['expected_return']:.2%}")
```

### JavaScript Client

```javascript
// Fetch comprehensive analysis
const response = await fetch('/api/portfolio/comprehensive-analysis', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    holdings: [
      {ticker: 'AAPL', weight: 0.3},
      {ticker: 'MSFT', weight: 0.3},
      {ticker: 'GOOGL', weight: 0.4}
    ],
    period: '2y'
  })
});

const data = await response.json();
console.log('Sharpe Ratio:', data.metrics.sharpe_ratio);
```

---

## References

1. **Markowitz, H.** (1952). Portfolio Selection. *Journal of Finance*
2. **Sharpe, W.** (1966). Mutual Fund Performance. *Journal of Business*
3. **Fama, E. & French, K.** (1993). Common risk factors in stock returns. *Journal of Financial Economics*
4. **Artzner, P. et al.** (1999). Coherent measures of risk. *Mathematical Finance*
5. **Black, F. & Litterman, R.** (1992). Global portfolio optimization. *Financial Analysts Journal*
6. **Ledoit, O. & Wolf, M.** (2004). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*
7. **Lopez de Prado, M.** (2016). Building diversified portfolios that outperform out-of-sample. *Journal of Portfolio Management*
8. **Hamilton, J.** (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*
9. **Meucci, A.** (2005). Risk and Asset Allocation. Springer
10. **Kelly, J.** (1956). A new interpretation of information rate. *Bell System Technical Journal*

---

## Performance Considerations

- **Caching**: Yahoo Finance data is cached for 1 hour
- **Concurrent fetching**: Multiple tickers fetched in parallel
- **Sampling**: Monte Carlo paths sampled to 50 for frontend visualization
- **Dimensionality**: PCA factors limited to min(5, n_assets-1)

---

## Future Enhancements

See [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) for planned features including:
- Deep reinforcement learning for portfolio optimization
- Alternative data integration (sentiment, ESG)
- Options analytics and hedging
- Full backtesting engine
- Tax-aware optimization
