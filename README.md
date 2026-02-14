# Portfolio Optimizer - Institutional-Grade Quantitative Analytics Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Flask-3.0+-green.svg" alt="Flask 3.0+">
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0+-orange.svg" alt="SQLAlchemy 2.0+">
  <img src="https://img.shields.io/badge/Neon%20DB-Ready-blueviolet.svg" alt="Neon DB Ready">
  <img src="https://img.shields.io/badge/Advanced%20Analytics-25%2B-brightgreen.svg" alt="25+ Analytics">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>

A comprehensive **Quantitative Portfolio Optimization Platform** featuring institutional-grade risk analytics, advanced statistical methods, and AI-powered investment insights. Built with Python Flask, featuring a responsive Tailwind CSS interface with 25+ interactive visualizations and comprehensive risk analysis.

## 🎯 What's New - Institutional-Grade Analytics

### Advanced Quantitative Analysis (NEW!)
- **25+ Interactive Charts** including Monte Carlo simulations, efficient frontier, rolling metrics
- **15+ Comprehensive Tables** with statistical breakdowns, risk metrics, factor exposures
- **10 Benchmark Comparisons** (S&P 500, FTSE 100, NASDAQ, MSCI World, etc.)
- **Statistical Tests** (Jarque-Bera, Ljung-Box, Shapiro-Wilk, Anderson-Darling, ARCH LM)
- **Distribution Analysis** (Normal, Student's t, Laplace) with AIC/BIC model selection
- **Extreme Value Theory** - Hill estimator, GPD parameters, Black Swan probability
- **Monte Carlo Simulation** - 5,000 paths, 1-year projections with confidence intervals

### Authentication & User Management (NEW!)
- **Registration/Login System** with admin approval workflow
- **Password Reset** via Google SMTP email
- **User Roles** - admin, user, pending approval
- **Admin Dashboard** - manage users, view statistics
- **Portfolio Scenario Saving** - save and retrieve analysis scenarios
- **Neon DB Optimization** - connection pooling, PgBouncer compatible

### Example Portfolio Showcase
- **Pre-computed Analysis** displayed on homepage for visitors
- **Complete Analytics** - risk, factors, correlations, stress tests
- **Interactive Charts** - all metrics visualized
- **Guest Mode** - full analysis without registration

---

## 🎯 Features

### Core Optimization Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Maximum Sharpe Ratio** | Tangency portfolio maximizing risk-adjusted returns | Growth-oriented investors |
| **Minimum Variance** | Lowest possible volatility portfolio | Risk-averse investors |
| **Risk Parity** | Equal risk contribution from all assets | True diversification seekers |
| **Mean-Variance** | Custom target return with minimum risk | Institutional mandates |

### 🤖 AI-Powered Document Analysis

Upload portfolio statements from any broker:
- **PDF Statements**: AI extracts holdings, quantities, and purchase dates
- **CSV Exports**: Automatic parsing of transaction history
- **Historical Tracking**: See how your portfolio risk evolved over time
- **Smart Recognition**: Works with statements from major brokers (Fidelity, Schwab, Vanguard, etc.)

### Technical Features
- **Real-time Data**: Yahoo Finance integration with intelligent caching
- **Interactive Visualizations**: Plotly efficient frontiers, Chart.js allocation charts
- **Risk Evolution Charts**: Track VaR, volatility, and Sharpe ratio over time
- **Flexible Database**: SQLite (default) or PostgreSQL (production/Neon DB)
- **RESTful API**: Complete API for programmatic access
- **Modern UI**: Responsive Tailwind CSS interface

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 - 3.13 (3.14+ may require compilation, see Arch Linux notes below)
- DeepSeek or OpenAI API key (optional, for AI features)

### Quick Install (Most Systems)

```bash
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer
python -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
./venv/bin/python -c "from app import app; from models.database import init_db; init_db(app)"
./venv/bin/flask --app app run --host=0.0.0.0 --port=5000 --debug
```

### Detailed Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer

# Check Python version (3.11-3.13 recommended)
python --version

# Create virtual environment
python -m venv venv

# Activate virtual environment (or use full paths, see below)
# bash/zsh: source venv/bin/activate
# fish:     . venv/bin/activate.fish
# Windows:  venv\Scripts\activate

# Install dependencies
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# Initialize database
./venv/bin/python -c "from app import app; from models.database import init_db; init_db(app)"

# Run the application
./venv/bin/flask --app app run --host=0.0.0.0 --port=5000 --debug
```

Visit `http://localhost:5000` in your browser.

### Virtual Environment Quick Reference

```bash
# Activate (bash/zsh on Linux/macOS)
source venv/bin/activate

# Activate (fish shell on Linux/macOS)
. venv/bin/activate.fish

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Deactivate (any shell)
deactivate

# Run Flask without activating (Linux/macOS)
./venv/bin/python app.py

# Or with flask CLI:
# ./venv/bin/python -m flask --app app run --host=0.0.0.0 --port=5000 --debug

# Run Flask without activating (Windows)
venv\Scripts\python app.py
```

---

## 📄 AI Document Upload Feature

### Supported Formats
- **PDF**: Brokerage statements, trade confirmations, portfolio summaries
- **CSV/Excel**: Transaction exports, position reports

### How It Works

1. **Upload** your portfolio statement via the web interface
2. **AI Extraction**: LLM parses the document to identify:
   - Ticker symbols and asset names
   - Share quantities and purchase prices
   - Transaction dates and types (buy/sell)
   - Current market values
3. **Historical Analysis**: Track how your portfolio composition and risk metrics evolved
4. **Visualization**: Interactive charts showing:
   - Risk metrics over time (VaR, Volatility, Drawdowns)
   - Asset allocation changes
   - Correlation shifts

### Example Workflow

```bash
# 1. Upload your Fidelity statement PDF
# 2. AI extracts: "Bought 100 AAPL @ $150 on 2023-01-15"
# 3. System fetches historical prices
# 4. Shows risk evolution: "Portfolio VaR increased 15% after tech sector addition"
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Application
SECRET_KEY=your-secret-key-here
DEBUG=True

# Database (SQLite default)
DB_TYPE=sqlite
SQLITE_DB_PATH=data/portfolio_optimizer.db

# OR PostgreSQL (for Neon DB)
DB_TYPE=postgresql
DATABASE_URL=postgresql://username:password@host.neon.tech/dbname?sslmode=require

# AI/LLM Configuration
LLM_API_KEY=your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# OR OpenAI
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4
```

### Neon DB Setup

1. Create account at [neon.tech](https://neon.tech)
2. Create a new project and database
3. Copy the connection string
4. Set `DATABASE_URL` in your `.env` file

---

## 🔬 Advanced Analytics Features

### Statistical Tests & Distribution Analysis

| Test | Purpose | Confidence Level |
|------|---------|-----------------|
| **Jarque-Bera** | Tests normality of returns | 95% |
| **Shapiro-Wilk** | Alternative normality test (more powerful for small samples) | 95% |
| **Ljung-Box** | Detects autocorrelation in returns | 95% |
| **Anderson-Darling** | Tests if sample follows specified distribution | 95% |
| **ARCH LM Test** | Detects volatility clustering (GARCH effects) | 95% |
| **Kolmogorov-Smirnov** | Goodness-of-fit for distribution models | 95% |

**Distribution Fitting:**
- **Normal Distribution** - Benchmark for comparison
- **Student's t** - Accounts for fat tails
- **Laplace Distribution** - Captures peak and tail behavior
- **AIC/BIC Model Selection** - Automatic best distribution selection

### Monte Carlo Simulation

- **5,000 Simulation Paths** - Robust forward projections
- **1-Year Forward Horizon** - Annual risk assessment
- **Percentile Bands** - 5th, 25th, 50th, 75th, 95th percentiles
- **Key Metrics:**
  - Probability of profit
  - Probability of 10%+ return
  - 95% confidence intervals
  - Worst/best case scenarios
  - Expected maximum drawdown

### Risk Analytics Suite

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Modified VaR** | Cornish-Fisher expansion adjusting for skewness/kurtosis | More accurate tail risk when returns are non-normal |
| **Kelly Criterion** | Optimal position sizing to maximize log wealth | Position sizing and leverage decisions |
| **Ulcer Index** | Downside risk focusing on drawdowns | Risk-adjusted performance in volatile markets |
| **Pain Ratio** | Return per unit of Ulcer Index | Alternative risk-adjusted measure |
| **Drawdown-at-Risk** | VaR for drawdowns instead of returns | Long-term capital preservation |
| **Tail Risk Analysis** | Extreme Value Theory, Hill estimator | Black swan event preparation |
| **Black Swan Probability** | Estimated from EVT | Risk of extreme events |
| **Expected Shortfall (CVaR)** | Average loss beyond VaR threshold | Deeper tail risk assessment |

### Volatility Analysis

- **Volatility Clustering (ARCH)** - Tests for volatility persistence
- **Volatility of Volatility** - Measures uncertainty in volatility
- **Volatility Persistence** - Half-life of volatility shocks
- **Rolling Volatility** - 30-day, 60-day, 90-day windows
- **Volatility Surface** - Time-dependent volatility structure

### Principal Component Analysis

- **Explained Variance Ratio** - PC1 and PC2 variance capture
- **Factor Loadings** - Asset sensitivity to principal components
- **Condition Number** - Multicollinearity detection
- **Effective Rank** - True dimensionality of portfolio

### Rolling Metrics with Confidence Bands

- **Rolling Mean Returns** - 63-day moving average with 95% CI
- **Rolling Volatility** - Dynamic risk measurement
- **Rolling Skewness/Kurtosis** - Distribution shape evolution
- **Rolling Sharpe Ratio** - Time-varying risk-adjusted performance
- **Rolling Beta vs S&P 500** - Market exposure tracking

### Advanced Optimization Methods

| Method | Description | Key Advantage |
|--------|-------------|---------------|
| **Black-Litterman** | Bayesian approach combining market equilibrium with views | Stable allocations with investor input |
| **Hierarchical Risk Parity** | Machine learning clustering for allocation | Works with correlated assets, no matrix inversion |
| **Ledoit-Wolf Shrinkage** | Improved covariance estimation | Better conditioning, reduced estimation error |
| **Factor Covariance** | PCA-based risk model | Handles high-dimensional portfolios |
| **Minimum Variance** | Global minimum variance portfolio | Lowest possible volatility |
| **Maximum Sharpe** | Tangency portfolio | Best risk-adjusted returns |

### Market Regime Detection

- **Hidden Markov Models** - Identify bull/bear markets automatically
- **Trend Following Filters** - Moving average-based regime classification
- **Regime-Dependent Allocation** - Different optimal portfolios per regime
- **Regime Switching Probabilities** - Transition matrix estimation

### Factor Analysis

- **Fama-French Models** - 3-factor, 5-factor, Carhart 4-factor
- **Statistical Factor Models** - PCA-based factor extraction
- **Factor Attribution** - Decompose returns into systematic and specific components
- **Factor Exposures** - Beta coefficients to each factor
- **Alpha Calculation** - Abnormal return after factor adjustment

### Stress Testing

- **2008 Financial Crisis** - Simulate market crash scenario
- **COVID-19 Crash** - Pandemic market stress test
- **Interest Rate Shock** - ±2% rate change impact
- **Inflation Shock** - High inflation environment
- **Geopolitical Crisis** - Custom scenario modeling

See [README_ADVANCED.md](README_ADVANCED.md) for detailed API documentation and mathematical formulas.

---

## 📈 Dashboard Visualizations

### Performance Charts (10+)

1. **Portfolio vs Benchmarks** - Multi-benchmark comparison with S&P 500, FTSE 100, NASDAQ, etc.
2. **Cumulative Returns** - Interactive time series with zoom and pan
3. **Efficient Frontier** - Monte Carlo simulation with 5,000 random portfolios
4. **Rolling Volatility** - 30/60/90-day volatility windows
5. **Rolling Sharpe Ratio** - Dynamic risk-adjusted performance
6. **Rolling Beta** - Market exposure over time vs S&P 500
7. **Drawdown Analysis** - Underwater chart with duration markers
8. **Monte Carlo Paths** - 5,000 simulation paths with percentile bands
9. **Return Distribution** - Histogram with fitted distributions overlay
10. **Correlation Heatmap** - Interactive correlation matrix

### Asset Analysis Charts (5)

11. **Asset Allocation** - Pie/donut chart with weights
12. **Risk Contribution** - Bar chart showing marginal risk contribution
13. **Sector Breakdown** - Sector allocation visualization
14. **Geographic Allocation** - Regional exposure pie chart
15. **Factor Exposure** - Radar chart of factor loadings

### Risk & Statistics Tables (15+)

**Performance Tables:**
- Performance Summary (returns, volatility, Sharpe, Sortino, Calmar)
- Monthly Returns Matrix (heatmap table)
- Calendar Year Returns
- Rolling Performance Metrics

**Risk Tables:**
- VaR Summary (95%, 99%, historical, parametric, modified)
- CVaR/Expected Shortfall Analysis
- Tail Risk Metrics (Hill estimator, GPD parameters)
- Stress Test Results (5 scenarios)
- Drawdown Statistics (worst 5, max duration)

**Statistical Tables:**
- Statistical Tests Summary (Jarque-Bera, Ljung-Box, Shapiro-Wilk, Anderson-Darling)
- Distribution Fitting Results (AIC/BIC comparison)
- Volatility Clustering Analysis (ARCH effects)
- Principal Component Analysis (PC1/PC2 loadings)

**Asset Detail Tables:**
- Asset Performance Metrics (each asset individually)
- Asset Risk Contributions
- Asset Correlation Matrix
- Sector/Industry Classification

**Factor Tables:**
- Fama-French Factor Exposures (3-factor, 5-factor)
- Factor Attribution Analysis
- Alpha/Beta Decomposition

**Optimization Tables:**
- Current vs Optimized Allocations
- Efficient Frontier Statistics
- Constraint Analysis

---

## 📊 Mathematical Framework

### Modern Portfolio Theory

The platform implements Harry Markowitz's seminal 1952 framework for portfolio selection:

Given $n$ assets with expected returns $\mu \in \mathbb{R}^n$ and covariance matrix $\Sigma \in \mathbb{R}^{n \times n}$, the optimization problem is:

$$
\begin{aligned}
\min_{w} \quad & w^T \Sigma w \\
\text{s.t.} \quad & \mathbf{1}^T w = 1 \\
& w \geq 0 \quad \text{(optional)}
\end{aligned}
$$

For the **Maximum Sharpe Ratio** portfolio:

$$
w^* = \arg\max_w \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
$$

### Risk Parity

Following Maillard, Roncalli & Teïletche (2010), the Risk Parity approach allocates such that each asset contributes equally to portfolio risk:

$$
\text{RC}_i = w_i \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}} = \frac{\sigma_p}{n}
$$

Where $\text{RC}_i$ is the risk contribution of asset $i$ and $\sigma_p$ is portfolio volatility.

### Value at Risk (VaR)

Parametric VaR calculation:

$$
\text{VaR}_\alpha = \mu_p - z_\alpha \cdot \sigma_p
$$

Where $z_\alpha$ is the quantile of the standard normal distribution.

---

## 📚 Research Foundations

### Seminal Works

1. **Markowitz, H. (1952)**. Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
   > "The process of selecting a portfolio may be divided into two stages: 
   > first, starting with observations and experience, ending with expectations 
   > about the performances of individual securities; second, starting with 
   > expectations about the performances of securities, ending with the 
   > choice of portfolio."

2. **Sharpe, W. F. (1966)**. Mutual Fund Performance. *The Journal of Business*, 39(1), 119-138.
   - Introduced the Sharpe Ratio for risk-adjusted performance measurement

3. **Black, F., & Litterman, R. (1992)**. Global Portfolio Optimization. 
   *Financial Analysts Journal*, 48(5), 28-43.
   - Bayesian approach combining market equilibrium with investor views

4. **Maillard, S., Roncalli, T., & Teïletche, J. (2010)**. The Properties 
   of Equally Weighted Risk Contribution Portfolios. *The Journal of Portfolio 
   Management*, 36(4), 60-70.
   - Mathematical foundation for Risk Parity strategies

5. **Lopez-Lira, A., & Tang, Y. (2023)**. Can ChatGPT Forecast Stock Price 
   Movements? Return Predictability and Large Language Models. *SSRN*.
   - Demonstrates LLM capabilities in financial forecasting

---

## 🛠️ Architecture

```
portfolio_optimizer/
├── app.py                 # Flask application entry point
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
│
├── models/               # Database models
│   └── database.py       # SQLAlchemy ORM definitions
│
├── services/             # Business logic
│   ├── market_data.py    # Yahoo Finance integration
│   ├── optimization.py   # Portfolio optimization engine
│   ├── ai_insights.py    # LLM-powered analysis
│   └── document_parser.py # PDF/CSV parsing with AI extraction
│
├── templates/            # Jinja2 HTML templates
│   ├── base.html         # Base layout
│   ├── index.html        # Dashboard
│   ├── optimize.html     # Advanced optimization
│   ├── analysis.html     # AI analysis interface
│   └── upload.html       # Document upload interface
│
└── static/               # Static assets
    ├── css/              # Custom styles
    └── js/               # Client-side scripts
```

### Design Patterns

- **Repository Pattern**: Abstracted data access through service classes
- **Strategy Pattern**: Pluggable optimization algorithms
- **Adapter Pattern**: Unified LLM interface supporting multiple providers
- **Factory Pattern**: Database connection management

---

## 🔌 API Reference

### Document Upload

#### POST `/api/upload`
Upload and parse portfolio document.

**Request:**
```bash
curl -X POST -F "file=@statement.pdf" \
             -F "broker=fidelity" \
             http://localhost:5000/api/upload
```

**Response:**
```json
{
  "success": true,
  "extracted_portfolio": {
    "holdings": [
      {"ticker": "AAPL", "shares": 100, "purchase_date": "2023-01-15", "cost_basis": 150.00},
      {"ticker": "MSFT", "shares": 50, "purchase_date": "2023-03-20", "cost_basis": 250.00}
    ],
    "cash": 5000.00
  },
  "confidence": 0.95
}
```

### Portfolio History

#### GET `/api/portfolios/<id>/history`
Get portfolio risk metrics over time.

**Response:**
```json
{
  "history": [
    {"date": "2023-01-15", "var_95": -1250.00, "volatility": 0.15, "sharpe": 1.2},
    {"date": "2023-02-15", "var_95": -1400.00, "volatility": 0.16, "sharpe": 1.1}
  ]
}
```

### Optimization

#### POST `/api/optimize`
Execute portfolio optimization.

**Request:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "method": "max_sharpe",
  "period": "2y",
  "risk_free_rate": 0.05,
  "constraints": {
    "allow_short": false,
    "max_position_size": 0.30
  }
}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_optimization.py
```

---

## 🚢 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```

### Render

1. Push code to GitHub
2. Create new Web Service on Render
3. Set environment variables
4. Deploy

### Heroku

```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
git push heroku main
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice. Always consult with a qualified financial advisor before making investment decisions.

Past performance is not indicative of future results. The optimization methods implemented are based on historical data and mathematical models that may not predict future market behavior.

---

## 📬 Contact

For questions or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com

---

## 🐧 Arch Linux / Python 3.14 Notes

### Python 3.14 Compatibility Issue

Arch Linux often ships with the latest Python version (3.14). Many packages (like pandas, numpy, scipy) don't have pre-built wheels for Python 3.14 yet, so they need to be compiled from source.

### Option 1: Use Python 3.12 (Recommended)

Install Python 3.12 from AUR or use pyenv:

```bash
# Using pyenv
sudo pacman -S pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Create venv with Python 3.12
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

### Option 2: Build from Source on Python 3.14

If you must use Python 3.14, install build dependencies first:

```bash
# 1. Install system build dependencies
sudo pacman -S --needed gcc gcc-fortran openblas lapack

# 2. Upgrade pip and install build tools
./venv/bin/pip install --upgrade pip wheel setuptools cython

# 3. Install numpy first (required by pandas/scipy)
./venv/bin/pip install numpy --no-build-isolation

# 4. Install pandas and scipy
./venv/bin/pip install pandas scipy --no-build-isolation

# 5. Install remaining packages
./venv/bin/pip install -r requirements.txt
```

### Fish Shell Users

If you're using **fish shell**, the activation script has different syntax:

```bash
# Wrong (bash syntax, won't work in fish):
source venv/bin/activate

# Correct (fish syntax):
. venv/bin/activate.fish
```

Or simply use the full path to run commands without activating:

```bash
# Run pip without activating
./venv/bin/pip install -r requirements.txt

# Run flask without activating  
./venv/bin/flask --app app run --host=0.0.0.0 --port=5000 --debug
```

### Quick Start for Arch Linux

```bash
cd portfolio_optimizer

# 1. Create venv
python -m venv venv

# 2. Install dependencies
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# If you get compilation errors with Python 3.14, see Option 2 above

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Initialize database
./venv/bin/python -c "
from app import app
with app.app_context():
    from models.database import db
    db.create_all()
    print('Database initialized!')
"

# 5. Run the application
./venv/bin/python app.py
# OR with flask CLI:
# ./venv/bin/python -m flask --app app run --host=0.0.0.0 --port=5000 --debug
```

---

<p align="center">
  Built with ❤️ and quantitative rigor
</p>
