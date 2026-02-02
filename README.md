# Portfolio Analysis & Management Pipeline

A Python-based research pipeline for stock screening, fundamental analysis, portfolio backtesting, weight optimization, and live portfolio monitoring.

> **Disclaimer:** This project is for **research and educational purposes only**. Nothing in this repository constitutes investment advice, a recommendation, or a solicitation to buy or sell any securities. Use at your own risk.

---

## Pipeline Overview

The project follows a sequential five-stage workflow. Each stage feeds into the next:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  1. Screening    │────▶│  2. Analysis     │────▶│  3. Backtesting      │
│  fundamentals.py │     │  Stock_Analysis  │     │  Portfolio_Backtest  │
│                  │     │  .ipynb          │     │  .ipynb              │
│  yfinance data   │     │  FactSet data    │     │  vartools backtest   │
└──────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                             │
                         ┌──────────────────┐     ┌──────────▼───────────┐
                         │  5. Monitoring   │◀────│  4. Deployment       │
                         │  Check_Portfolio │     │  Portfolio_Weights   │
                         │  .py             │     │  .ipynb              │
                         │  IBKR API        │     │  vartools optimizer  │
                         └──────────────────┘     └──────────────────────┘
```

---

## Stages

### 1. Screening — `fundamentals.py`

Performs an initial screening of a list of tickers using data from Yahoo Finance.

| What it does | How |
|---|---|
| Fetches balance sheet, income statement, cash flow | `yfinance` API |
| Computes key ratios (margins, ROE, FCF/Sales, current ratio, solvency) | Pandas operations |
| Downloads price data and computes annualized volatility | Trailing 1-year window |
| Generates a combined dashboard per ticker | Matplotlib, saved as PNG |
| Processes tickers in parallel | `ProcessPoolExecutor` |

**Output:** A `Financials/<TICKER>/` folder with a `<TICKER>_overview.png` dashboard for each ticker.

**Usage:**
```bash
python fundamentals.py
```
Edit the `tickers` list in `main()` to screen different stocks.

---

### 2. Fundamental Analysis — `Stock_Analysis.ipynb`

Takes the best candidates from screening and performs deep fundamental analysis using institutional-grade quarterly LTM data from FactSet.

| What it does | How |
|---|---|
| Loads balance sheet, income statement, cash flow, and ratio analysis from Excel | FactSet exports in `FactSet/<TICKER>/` |
| Visualizes profitability, valuation, debt structure, margins, EBITDA, FCF | Multiple chart types |
| Pulls analyst price targets and recommendations | `yfinance` API |
| Computes a composite stock score across 5 pillars | Custom scoring model |
| Ranks all tickers by final score | Weighted composite |

**Scoring Model:**

| Pillar | Weight | Key Metrics |
|---|---|---|
| Profitability | 35% | Gross/Operating/Net Margin, ROE, ROIC |
| Growth | 25% | Sales, EBITDA, EBIT, EPS |
| Cash Flow | 15% | Free Cash Flow, FCF Margin |
| Financial Strength | 15% | Current Ratio, Debt/EBITDA, Interest Coverage |
| Valuation | 10% | P/E, P/B |

Each metric is scored by combining its historical percentile rank (level) with a weighted trend measure (momentum over 1Y, 2Y, 3Y).

**Prerequisites:** Download quarterly LTM data from FactSet and place it in:
```
FactSet/
├── AAPL/
│   ├── bs.xlsx
│   ├── ist.xlsx
│   ├── cf.xlsx
│   └── ra.xlsx
├── MSFT/
│   └── ...
```

**Usage:** Run all cells sequentially. Set the `ticker` variable for single-stock analysis, and the `tickers` list for batch scoring.

---

### 3. Backtesting — `Portfolio_Backtest.ipynb`

Once the best stocks are selected, this notebook backtests six different asset allocation strategies over a 5-year period to evaluate performance.

| Strategy | Description |
|---|---|
| Min Variance | Minimizes portfolio variance |
| Sharpe | Maximizes the Sharpe ratio |
| Semivariance | Minimizes downside variance |
| Omega | Maximizes the Omega ratio |
| Min CVaR | Minimizes Conditional Value at Risk |
| MCC | Mean-Correlation Criterion |

The backtest is **dynamic** — portfolios are rebalanced at a configurable interval (default: quarterly) using a rolling optimization window.

**What it produces:**

- Cumulative value evolution vs. benchmark (SPY)
- Rolling annualized volatility per strategy
- Rolling mean correlation of the asset universe
- Performance metrics table with heatmap styling:

| Metric | Description |
|---|---|
| Return | Annualized return |
| Volatility | Annualized standard deviation |
| Sharpe | Risk-adjusted return |
| Sortino | Downside risk-adjusted return |
| Omega | Upside/downside volatility ratio |
| Alpha | Excess return over CAPM |
| Beta | Systematic risk |

- Return distribution histogram and daily return bar chart for the selected strategy
- VaR and CVaR at 99% confidence

**Usage:** Set the `tickers` list to your selected stocks, adjust `months` for rebalancing frequency, and run all cells.

---

### 4. Weight Deployment — `Portfolio_Weights.ipynb`

After choosing the best strategy and rebalancing period from the backtest, this notebook computes the current optimal weights using the most recent data.

| What it does | How |
|---|---|
| Downloads the latest price window matching the rebalancing period | `yfinance` |
| Runs portfolio optimization for the chosen strategy | `vartools.OptimizePortfolioWeights` |
| Displays target weight allocation | Pie/bar chart |
| Computes exact share quantities to buy/sell | `vartools.rebalance_stocks` |

**Output:** A table showing per-ticker original weight, optimal weight, and number of shares to trade given your capital.

**Usage:** Set `tickers`, `portfolio_value`, and `w_original` (zeros for initial allocation, current weights for rebalancing). Run all cells.

---

### 5. Portfolio Monitoring — `Check_Portfolio.py`

A standalone script that connects to Interactive Brokers TWS to snapshot the current portfolio.

| What it does | How |
|---|---|
| Connects to TWS/Gateway via localhost | IBKR API (`ibapi`) |
| Retrieves all positions with P&L | Account updates callback |
| Computes portfolio weights and unrealized P&L % | Pandas |
| Saves a timestamped Excel snapshot | `Portfolio_History/Portfolio_Checkpoint_DD-MM-YYYY.xlsx` |

**Prerequisites:**
- An [Interactive Brokers](https://www.interactivebrokers.com/) account
- TWS or IB Gateway running with API connections enabled
- The `ibapi` Python package installed and configured

For setup instructions on how to enable the API and install `ibapi`, check the official Interactive Brokers video tutorial:
> **[https://www.youtube.com/watch?v=_hjgBid_Rcc]**

**Configuration:** Create a `.env` file in the project root:
```env
IBKR_HOST=127.0.0.1
IBKR_PORT=7496 or 7497
IBKR_CLIENT_ID= your choice
```

**Usage:**
```bash
python Check_Portfolio.py
```

---

## vartools

Much of the heavy lifting for portfolio optimization and backtesting is handled by `vartools`, a custom Python library built for this workflow.

Check out the library and its documentation here:
> **[https://github.com/LuisMB09/vartools]**

---

## Installation

```bash
git clone <repo-url>
cd Portfolio
pip install -r requirements.txt
```

> **Note:** `ibapi` is not included in `requirements.txt` as it requires a separate installation from Interactive Brokers. It is only needed for `Check_Portfolio.py`.

---

## Project Structure

```
Portfolio/
├── fundamentals.py              # Stage 1: Stock screening
├── Stock_Analysis.ipynb         # Stage 2: Fundamental analysis & scoring
├── Portfolio_Backtest.ipynb     # Stage 3: Strategy backtesting
├── Portfolio_Weights.ipynb      # Stage 4: Weight optimization & rebalancing
├── Check_Portfolio.py           # Stage 5: Live portfolio monitoring (IBKR)
├── requirements.txt             # Python dependencies
├── .env                         # IBKR connection config (not tracked)
├── .gitignore
├── FactSet/                     # FactSet Excel data (not tracked)
├── Financials/                  # Generated screening dashboards (not tracked)
└── Portfolio_History/           # IBKR portfolio snapshots (not tracked)
```
