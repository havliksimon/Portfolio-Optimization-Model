# Portfolio Optimizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Flask-3.0+-green.svg" alt="Flask 3.0+">
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0+-orange.svg" alt="SQLAlchemy 2.0+">
  <img src="https://img.shields.io/badge/25%2B%20Analytics-Institutional%20Grade-brightgreen.svg" alt="25+ Analytics">
  <img src="https://img.shields.io/badge/Free%20Deployment-Koyeb%2FRender%20%2B%20Neon-blueviolet.svg" alt="Free Deployment">
</p>

<p align="center">
  <em>Where Modern Portfolio Theory meets computational precision.</em>
</p>

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-repo/portfolio-optimizer.git
cd portfolio-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
cp .env.example .env
python app.py
```

**Visit:** http://localhost:5000

---

## Documentation

| Document | Description |
|----------|-------------|
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Free cloud deployment on Koyeb/Render + Neon DB |
| **[ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)** | Complete feature list |
| **This README** | Technical overview and mathematical foundations |

---

## Overview

A comprehensive quantitative portfolio optimization platform implementing cutting-edge financial mathematics and statistical methods. From Markowitz's mean-variance optimization to sophisticated tail risk analysis using Extreme Value Theory, this platform bridges academic rigor with practical implementation.

### What Makes This Different

While many portfolio tools offer basic metrics, this platform dives deep into the distributional properties of returns, implementing methods typically reserved for institutional quant desks:

- **Distribution Fitting with AIC/BIC Selection** - Rather than assuming normality, we fit Normal, Student's t, and Laplace distributions, selecting the best model via information criteria
- **Extreme Value Theory** - Hill estimators and Generalized Pareto Distribution for understanding tail risks beyond VaR
- **Monte Carlo Simulation** - 5,000 path simulations with full percentile bands, not just point estimates
- **Statistical Test Suite** - Jarque-Bera, Ljung-Box, Shapiro-Wilk, Anderson-Darling, ARCH-LM tests
- **Principal Component Analysis** - Understanding the true dimensionality of portfolio risk

---

## Mathematical Foundations

### Modern Portfolio Theory

The platform implements Harry Markowitz's 1952 framework through quadratic programming:

$$
\begin{aligned}
\min_{w} \quad & w^T \Sigma w \\
\text{s.t.} \quad & \mathbf{1}^T w = 1 \\
& w \geq 0 \quad \text{(no-short constraints)}
\end{aligned}
$$

For maximum Sharpe ratio optimization:

$$
w^* = \arg\max_w \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
$$

### Risk Parity

Following Maillard, Roncalli & Teïletche (2010), the Risk Parity portfolio solves:

$$
\text{RC}_i = w_i \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}} = \frac{\sigma_p}{n}
$$

### Extreme Value Theory

For tail risk estimation, we implement the Hill estimator for the tail index $\xi$:

$$
\hat{\xi}_k = \frac{1}{k} \sum_{i=1}^k \log\frac{X_{n-i+1,n}}{X_{n-k,n}}
$$

Where $X_{i,n}$ are the order statistics. This feeds into GPD parameter estimation for VaR/CVaR beyond historical simulation limits.

---

## Analytics Engine

### Statistical Tests

| Test | Purpose | Implementation |
|------|---------|----------------|
| **Jarque-Bera** | Normality via skewness/kurtosis | $\chi^2$ test with 2 df |
| **Shapiro-Wilk** | Powerful normality test (small samples) | W-statistic via order statistics |
| **Ljung-Box** | Serial correlation in returns | $Q = n(n+2)\sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k}$ |
| **Anderson-Darling** | Distribution goodness-of-fit | $A^2 = -n - S$ |
| **ARCH-LM** | Volatility clustering detection | LM test on squared residuals |

### Distribution Analysis

Rather than assuming normality, we fit and compare:

```python
# Normal distribution - baseline comparison
# Student's t - captures fat tails via degrees of freedom
# Laplace - double exponential for leptokurtic returns
```

Model selection via:
- **AIC**: $-2\ln(L) + 2k$
- **BIC**: $-2\ln(L) + k\ln(n)$

### Monte Carlo Simulation

```
5,000 independent paths
Geometric Brownian Motion with drift
Confidence intervals: 5th, 25th, 50th, 75th, 95th percentiles
Worst-case scenario analysis (0.5% tail)
```

---

## Architecture

```
portfolio_optimizer/
├── app.py                 # Flask application with 10,000+ lines of quant logic
├── config.py             # 12-Factor configuration management
├── requirements.txt      # Dependency management
│
├── models/               # Database ORM
│   └── database.py       # SQLAlchemy with PostgreSQL/SQLite
│
├── services/             # Core business logic
│   ├── market_data.py    # Yahoo Finance integration with intelligent caching
│   ├── optimization.py   # CVXPY-based convex optimization
│   ├── risk_analytics.py # VaR, CVaR, drawdown analysis
│   ├── advanced_statistics.py  # Distribution fitting, hypothesis testing
│   ├── advanced_risk.py  # EVT, tail risk, Kelly criterion
│   ├── factor_models.py  # Fama-French 3/5 factor decomposition
│   ├── black_litterman.py # Bayesian portfolio optimization
│   ├── regime_detection.py # Hidden Markov Models for market regimes
│   └── cache_service.py  # Multi-layer Redis + in-memory caching
│
├── templates/            # Jinja2 + Tailwind CSS
│   ├── base.html         # Glassmorphism UI with custom cursor
│   └── index.html        # Real-time analytics dashboard
│
└── static/               # Premium UX assets
    ├── css/advanced.css  # Custom cursor, animations, glass effects
    └── js/cursor.js      # Interactive cursor with trail
```

---

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer
python -m venv venv
./venv/bin/pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Initialize database
./venv/bin/python -c "from app import app; from models.database import init_db; init_db(app)"

# Run
./venv/bin/flask --app app run --host=0.0.0.0 --port=5000 --debug
```

### Free Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for zero-cost deployment on:
- **Koyeb** or **Render** (application hosting)
- **Neon** (serverless PostgreSQL)
- **Upstash** (Redis caching)

**Total cost: $0/month** with smart caching to minimize database hits.

---

## Key Features

### Portfolio Optimization Methods

| Method | Mathematical Approach | Best For |
|--------|----------------------|----------|
| **Maximum Sharpe** | Quadratic programming with risk-free rate | Growth portfolios |
| **Minimum Variance** | Global minimum variance via convex optimization | Capital preservation |
| **Risk Parity** | Equal risk contribution (Newton-Raphson solver) | True diversification |
| **Black-Litterman** | Bayesian mixing of equilibrium & views | Incorporating forecasts |
| **Hierarchical RP** | Machine learning clustering + recursive bisection | Correlated assets |

### Risk Metrics Suite

**Standard Metrics:**
- Sharpe, Sortino, Calmar, Omega ratios
- Treynor, Information ratios
- Beta, Jensen's Alpha

**Advanced Metrics:**
- Modified VaR (Cornish-Fisher expansion)
- Conditional VaR / Expected Shortfall
- CVaR optimization
- Drawdown-at-Risk
- Ulcer Index & Pain Ratio
- Kelly Criterion position sizing

**Tail Risk:**
- Hill Estimator for tail index
- GPD shape parameter estimation
- Black Swan probability (EVT-based)

### Factor Analysis

```
Fama-French 3-Factor:
r - r_f = α + β_m(r_m - r_f) + β_s SMB + β_v HML + ε

Fama-French 5-Factor (adds):
+ β_r RMW (Profitability)
+ β_c CMA (Investment)

Statistical Factor Models:
- PCA-based factor extraction
- Factor attribution analysis
```

---

## Research Foundations

1. **Markowitz, H. (1952)**. Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.

2. **Sharpe, W. F. (1966)**. Mutual Fund Performance. *The Journal of Business*, 39(1), 119-138.

3. **Black, F., & Litterman, R. (1992)**. Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28-43.

4. **Maillard, S., Roncalli, T., & Teïletche, J. (2010)**. The Properties of Equally Weighted Risk Contribution Portfolios. *The Journal of Portfolio Management*, 36(4), 60-70.

5. **Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997)**. Modelling Extremal Events. *Springer*.

---

## Technical Stack

**Backend:**
- Python 3.11+ with Flask
- SQLAlchemy 2.0 (PostgreSQL/SQLite)
- CVXPY for convex optimization
- NumPy/SciPy/Pandas for numerical computing
- Statsmodels for statistical testing

**Frontend:**
- Tailwind CSS with custom glassmorphism
- Plotly/Chart.js for interactive visualizations
- Custom cursor with trail effect
- KaTeX for math rendering

**Infrastructure:**
- Multi-layer caching (Redis + in-memory)
- Connection pooling for serverless databases
- Graceful degradation patterns

---

## API Reference

### Portfolio Analysis

```bash
POST /api/portfolio/comprehensive-analysis
Content-Type: application/json

{
  "holdings": [
    {"ticker": "AAPL", "weight": 0.25},
    {"ticker": "MSFT", "weight": 0.25},
    {"ticker": "GOOGL", "weight": 0.25},
    {"ticker": "AMZN", "weight": 0.25}
  ],
  "period": "2y",
  "portfolio_value": 100000
}
```

Returns complete analysis including:
- Risk metrics (VaR, CVaR, Sharpe, etc.)
- Statistical tests (Jarque-Bera, Ljung-Box, etc.)
- Monte Carlo simulation results
- Factor exposures
- Stress test scenarios
- Distribution fitting results

---

## Configuration

### Environment Variables

```env
# Required
SECRET_KEY=your-secret-key

# Database (choose one)
DATABASE_URL=postgresql://...  # Neon/PostgreSQL
SQLITE_DB_PATH=data/app.db     # SQLite (local dev)

# Optional - for AI features
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com/v1

# Caching (recommended for production)
REDIS_URL=rediss://...upstash.io
ENABLE_CACHE=true
CACHE_TTL=3600
```

---

## Testing

```bash
# Run test suite
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific modules
pytest tests/test_optimization.py -v
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with computational precision.
</p>
