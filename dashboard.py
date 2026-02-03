import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import vartools as vt
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

np.random.seed(42)
plt.rcParams['figure.facecolor'] = 'lightgray'

STRATEGY_COLORS = {
    'Semivariance': 'navy',
    'Min Variance': 'skyblue',
    'Sharpe': 'cornflowerblue',
    'Omega': 'black',
    'Min CVaR': 'darkorange',
    'MCC': 'firebrick',
}

STRATEGY_OPT_MAP = {
    'Semivariance': 'opt_min_semivar',
    'Min Variance': 'opt_min_var',
    'Sharpe': 'opt_max_sharpe',
    'Omega': 'opt_max_omega',
    'Min CVaR': 'opt_min_cvar',
    'MCC': 'opt_mcc',
}

# ── Helper Functions ──────────────────────────────────────────────────────────

def rolling_mean_corr(rt: pd.DataFrame, window: int) -> pd.Series:
    mean_corr = []
    for i in range(window, len(rt) + 1):
        corr = rt.iloc[i - window:i].corr().values
        mask = ~np.eye(corr.shape[0], dtype=bool)
        mean_corr.append(corr[mask].mean())
    return pd.Series(mean_corr, index=rt.index[window - 1:])

def var(returns):
    return np.percentile(returns, 1)

def cvar(returns):
    v = np.percentile(returns, 1)
    return returns[returns < v].mean()

def get_metrics(history: pd.DataFrame, rf: float, months: int, benchmark_data: pd.DataFrame):
    n_days = round(len(benchmark_data) / round(len(benchmark_data) / 252 / (months / 12)), 0)
    filtered_benchmark_data = benchmark_data.iloc[int(n_days):]

    daily_rets = history.pct_change().dropna()
    benchmark_rets = filtered_benchmark_data.pct_change().dropna()

    X = benchmark_rets.values.reshape(-1, 1)
    y = daily_rets
    betas = [LinearRegression().fit(X, y.values[:, i]).coef_[0] for i in range(y.shape[1])]

    rend_prom = daily_rets.mean() * 252
    std__ = daily_rets.std() * np.sqrt(252)
    RS = (rend_prom - rf) / std__
    downside = daily_rets[daily_rets < 0].fillna(0).std() * np.sqrt(252)
    upside = daily_rets[daily_rets > 0].fillna(0).std() * np.sqrt(252)
    omega = upside / downside
    sortino = (rend_prom - rf) / downside

    benchmark_ret = benchmark_rets.mean().iloc[0] * 252
    beta_series = pd.Series(betas, index=daily_rets.columns)
    alpha = rend_prom - (rf + beta_series * (benchmark_ret - rf))

    metrics = pd.DataFrame([rend_prom, std__, RS, downside, upside, omega, sortino],
                           index=['Rend', 'Vol', 'Sharpe', 'Downside', 'Upside', 'Omega', 'Sortino'])
    metrics.loc['Beta'] = betas
    metrics.loc['Alpha'] = alpha
    return metrics

def get_benchmark_metrics(rt_benchmark_test: pd.DataFrame, rf: float):
    benchmark_ret = rt_benchmark_test.mean() * 252
    benchmark_vol = rt_benchmark_test.std() * np.sqrt(252)
    benchmark_sharpe = (benchmark_ret - rf) / benchmark_vol
    benchmark_down_risk = np.minimum(rt_benchmark_test, 0).std() * np.sqrt(252)
    benchmark_sortino = (benchmark_ret - rf) / benchmark_down_risk
    benchmark_upside_risk = np.maximum(rt_benchmark_test, 0).std() * np.sqrt(252)
    benchmark_omega = benchmark_upside_risk / benchmark_down_risk

    summary = pd.DataFrame({
        'Rend': benchmark_ret,
        'Vol': benchmark_vol,
        'Sharpe': benchmark_sharpe,
        'Downside': benchmark_down_risk,
        'Upside': benchmark_upside_risk,
        'Omega': benchmark_omega,
        'Sortino': benchmark_sortino,
        'Alpha': 0,
        'Beta': 1
    })
    summary.set_index(pd.Index(['Benchmark']), inplace=True)
    return summary.T

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("Portfolio Dashboard")
st.markdown("""
This dashboard allows you to backtest and compare multiple portfolio optimization strategies
using historical market data. Input your tickers, capital, and risk parameters to analyze
performance metrics, rolling risk indicators, and optimal portfolio weights across strategies
such as Minimum Variance, Maximum Sharpe, Semivariance, Omega, CVaR, and MCC.

**Disclaimer:** This tool is intended for **research, analysis, and educational purposes only**.
It does not constitute financial advice, and should not be used as the sole basis for any
investment decision. Past performance does not guarantee future results.

---
""")

# ── Sidebar Inputs ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Parameters")
    tickers_input = st.text_input("Tickers (comma-separated)", "NVDA,AMZN,AVGO,PG,V,RL,GLD")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    rf = st.number_input("Risk-Free Rate", value=0.0375, format="%.4f")
    capital = st.number_input("Capital ($)", value=1_000_000.0, step=10_000.0, format="%.2f")
    months = st.number_input("Rebalancing Period (months)", value=3, min_value=1, max_value=24, step=1)
    run = st.button("Run Analysis", type="primary")

if len(tickers) < 2:
    st.error("Please enter at least 2 tickers.")
    st.stop()

# Persist analysis results in session_state so strategy selection doesn't re-run everything
if run:
    st.session_state["run_params"] = {
        "tickers": tickers, "rf": rf, "capital": capital, "months": months
    }

if "run_params" not in st.session_state:
    st.info("Set your parameters in the sidebar and click **Run Analysis**.")
    st.stop()

# Use stored params
params = st.session_state["run_params"]
tickers = params["tickers"]
rf = params["rf"]
capital = params["capital"]
months = params["months"]

# ── Download & Compute (cached in session_state) ─────────────────────────────

if run or "price" not in st.session_state:
    tickers = [t.upper() for t in tickers]
    benchmark = "SPY"
    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    with st.spinner("Downloading data..."):
        try:
            price = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"][tickers]
            benchmark_price = yf.download(benchmark, start=start_date, end=end_date, progress=False)["Close"]
            if price.empty or benchmark_price.empty:
                st.error("Could not download data. Check your tickers and try again.")
                st.stop()
        except Exception as e:
            st.error(f"Data download failed: {e}")
            st.stop()

    rt = price.pct_change().dropna()

    n_days = round(len(benchmark_price) / round(len(benchmark_price) / 252 / (months / 12)), 0)
    benchmark_test = benchmark_price.iloc[int(n_days):]
    rt_benchmark_test = benchmark_test.pct_change().fillna(0)

    with st.spinner("Running backtesting simulation..."):
        history = vt.DynamicBacktesting(price, benchmark_price, capital=capital, rf=rf, months=months, alpha=95).simulation()
        history_rets = history.pct_change().dropna()

    cumulative_benchmark_returns = (1 + rt_benchmark_test).cumprod().squeeze()
    cumulative_money_benchmark = capital * cumulative_benchmark_returns.values

    rolling_corr = rolling_mean_corr(rt, window=60)
    rolling_std_portfolios = history_rets.rolling(window=60).std().dropna() * np.sqrt(252)
    rolling_std_benchmark = rt_benchmark_test.rolling(window=60).std().dropna() * np.sqrt(252)

    benchmark_metrics = get_benchmark_metrics(rt_benchmark_test, rf)
    portfolio_metrics = get_metrics(history, rf, months, benchmark_price)
    all_metrics = pd.concat([benchmark_metrics, portfolio_metrics], axis=1)

    # Optimization data (last n_days window)
    opt_data = price.iloc[-int(n_days):]
    opt_rt = opt_data.pct_change().dropna()
    opt_rt_benchmark = benchmark_price.iloc[-int(n_days):].pct_change().dropna()

    # Precompute weights and simulations for all strategies
    all_weights = {}
    all_simulations = {}
    with st.spinner("Optimizing weights and running simulations for all strategies..."):
        optimizer = vt.OptimizePortfolioWeights(opt_rt, risk_free=rf)
        for strategy, method_name in STRATEGY_OPT_MAP.items():
            method = getattr(optimizer, method_name)
            if method_name in ("opt_min_semivar", "opt_max_omega"):
                w = method(opt_rt_benchmark)
            elif method_name in ("opt_min_cvar", "opt_mcc"):
                w = method(95)
            else:
                w = method()
            all_weights[strategy] = w
            all_simulations[strategy] = vt.simulate_portfolio(opt_data, w, int(n_days))

    # Store everything
    st.session_state.update({
        "price": price, "benchmark_price": benchmark_price,
        "rt": rt, "n_days": n_days,
        "rt_benchmark_test": rt_benchmark_test,
        "history": history, "history_rets": history_rets,
        "cumulative_money_benchmark": cumulative_money_benchmark,
        "rolling_corr": rolling_corr,
        "rolling_std_portfolios": rolling_std_portfolios,
        "rolling_std_benchmark": rolling_std_benchmark,
        "all_metrics": all_metrics,
        "opt_data": opt_data, "opt_rt": opt_rt, "opt_rt_benchmark": opt_rt_benchmark,
        "all_weights": all_weights, "all_simulations": all_simulations,
    })

# Retrieve from session_state
price = st.session_state["price"]
benchmark_price = st.session_state["benchmark_price"]
rt = st.session_state["rt"]
n_days = st.session_state["n_days"]
rt_benchmark_test = st.session_state["rt_benchmark_test"]
history = st.session_state["history"]
history_rets = st.session_state["history_rets"]
cumulative_money_benchmark = st.session_state["cumulative_money_benchmark"]
rolling_corr = st.session_state["rolling_corr"]
rolling_std_portfolios = st.session_state["rolling_std_portfolios"]
rolling_std_benchmark = st.session_state["rolling_std_benchmark"]
all_metrics = st.session_state["all_metrics"]
opt_data = st.session_state["opt_data"]
opt_rt = st.session_state["opt_rt"]
opt_rt_benchmark = st.session_state["opt_rt_benchmark"]
all_weights = st.session_state["all_weights"]
all_simulations = st.session_state["all_simulations"]

# ── Annual Expected Return & Volatility ───────────────────────────────────────

st.header("Annual Expected Return & Annualized Volatility")
stats_df = pd.DataFrame({
    'Annual Return': rt.mean() * 252,
    'Annual Volatility': rt.std() * np.sqrt(252),
})
st.dataframe(stats_df.T.style.format("{:.2%}"))

# ── Correlation Heatmap ──────────────────────────────────────────────────────

st.header("Correlation Heatmap")
corr = rt.corr()
plt.figure(figsize=(6, 6))
sns.heatmap(corr, annot=True, cmap="Blues", vmin=-1, vmax=1, linewidths=0.5, linecolor="black")
plt.title("Correlation Heatmap of Stock Returns", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

mask = ~np.eye(corr.values.shape[0], dtype=bool)
avg_corr = corr.values[mask].mean()
st.metric("Average Pairwise Correlation", f"{avg_corr:.4f}")

# ── Portfolio Evolution ──────────────────────────────────────────────────────

st.header("Dynamic Backtesting")

st.subheader("Portfolio Evolution")

# Plotly version
fig_plotly = go.Figure()
fig_plotly.add_trace(go.Scatter(x=rt_benchmark_test.index, y=cumulative_money_benchmark, name="Benchmark", line=dict(color="gray")))
for strategy, color in STRATEGY_COLORS.items():
    fig_plotly.add_trace(go.Scatter(x=rt_benchmark_test.index, y=history[strategy], name=strategy, line=dict(color=color)))
fig_plotly.update_layout(
    title="Portfolio Value Over Time",
    xaxis_title="Date", yaxis_title="Value ($)",
    legend=dict(bgcolor="white", bordercolor="black", borderwidth=1, font=dict(color="black")),
    template="plotly_white",
    height=500,
    plot_bgcolor="whitesmoke",
    paper_bgcolor="lightgray",
    font=dict(color="black"),
    title_font=dict(color="black"),
    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
)
# Show strategy name next to each line on hover
fig_plotly.update_traces(mode="lines", hovertemplate="%{fullData.name}<br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>")
st.plotly_chart(fig_plotly, use_container_width=True)


st.write(f"**Benchmark final value:** ${cumulative_money_benchmark[-1]:,.2f}")
for col in history.columns:
    st.write(f"**{col} final value:** ${history[col].iloc[-1]:,.2f}")

# Rolling correlation
st.subheader("Rolling Mean Correlation (60-day)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(rolling_corr.index, rolling_corr.values, color="royalblue")
ax.set_title("Rolling Mean Correlation of Stock Returns")
ax.set_ylabel("Mean Correlation")
ax.grid(True, linestyle="--", color="gray", alpha=0.7)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Rolling volatility
st.subheader("Rolling Annualized Volatility (60-day)")
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=rolling_std_benchmark.index, y=rolling_std_benchmark.squeeze(), name="Benchmark", line=dict(color="gray")))
for strategy, color in STRATEGY_COLORS.items():
    fig_vol.add_trace(go.Scatter(x=rolling_std_portfolios.index, y=rolling_std_portfolios[strategy], name=strategy, line=dict(color=color)))
fig_vol.update_layout(
    title="Rolling Annualized Volatility",
    xaxis_title="Date", yaxis_title="Annualized Volatility",
    legend=dict(bgcolor="white", bordercolor="black", borderwidth=1, font=dict(color="black")),
    template="plotly_white",
    height=500,
    plot_bgcolor="whitesmoke",
    paper_bgcolor="lightgray",
    font=dict(color="black"),
    title_font=dict(color="black"),
    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
)
fig_vol.update_traces(mode="lines", hovertemplate="%{fullData.name}<br>Date: %{x}<br>Vol: %{y:.4f}<extra></extra>")
st.plotly_chart(fig_vol, use_container_width=True)

# ── Performance Metrics ──────────────────────────────────────────────────────

st.header("Performance Metrics")

st.subheader("Metrics (actual values)")
st.dataframe(all_metrics.style.format("{:.4f}"))

st.subheader("Metrics (color-coded reference)")
lower_is_better = ["Vol", "Downside", "Beta"]

def rowwise_gradient(row):
    if row.name in lower_is_better:
        return pd.Series(row.max() - row, index=row.index)
    return row

styled = all_metrics.copy().apply(rowwise_gradient, axis=1)
st.dataframe(styled.style.background_gradient(cmap="RdYlGn", axis=1).format("{:.4f}"))
st.caption("The colored table is for visual reference only. Refer to the table above for actual values.")

# ── VaR & C-VaR ─────────────────────────────────────────────────────────────

st.header("VaR & C-VaR (99% confidence, 1-day)")

var_data = {}
for col in list(history.columns) + ["Benchmark"]:
    rets = history_rets[col] if col in history_rets.columns else rt_benchmark_test
    if isinstance(rets, pd.DataFrame):
        rets = rets.squeeze()
    var_data[col] = {"VaR (1%)": var(rets), "C-VaR (1%)": cvar(rets)}

var_df = pd.DataFrame(var_data)
st.dataframe(var_df.style.format("{:.2%}"))

# ── Strategy Selection ───────────────────────────────────────────────────────

st.header("Choose Your Strategy")
strategies = list(STRATEGY_COLORS.keys())
chosen = st.selectbox("Select the strategy you prefer:", strategies)

if chosen:
    # Histogram of returns
    st.subheader(f"Returns Distribution — {chosen}")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(history_rets[chosen], color="navy", alpha=0.1, kde=True, bins=50, edgecolor=None, ax=ax)
    ax.axvline(x=0, color="red", linestyle="--", label="Zero")
    ax.set_title(f"Returns Distribution of the {chosen} Portfolio")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Positive / Negative bar plot
    st.subheader(f"Daily Returns — {chosen}")
    rets = history_rets[chosen]
    positive = (rets > 0).sum()
    negative = (rets < 0).sum()
    total = positive + negative
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(rets.index[rets > 0], rets[rets > 0], color="green", label=f"Positive: {positive/total:.2%}")
    ax.bar(rets.index[rets < 0], rets[rets < 0], color="red", label=f"Negative: {negative/total:.2%}")
    ax.hlines(0, rets.index[0], rets.index[-1], colors="black", linestyles="dashed")
    ax.set_title(f"Daily Returns of the {chosen} Portfolio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Optimize Weights ─────────────────────────────────────────────────────

    st.header(f"Optimal Weights — {chosen}")

    weights = all_weights[chosen]

    vt.plot_weights(tickers, weights)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    weights_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    st.dataframe(weights_df.style.format({"Weight": "{:.4f}"}))

    # ── Rebalancing ──────────────────────────────────────────────────────────

    st.header("Rebalancing")

    w_original = np.zeros(len(tickers))
    rebalance_df = vt.rebalance_stocks(
        w_original=w_original,
        target_weights=weights,
        data=price,
        stocks=tickers,
        portfolio_value=capital,
    )
    st.table(rebalance_df)

    # ── Portfolio Simulation ─────────────────────────────────────────────────

    st.header("Simulated Portfolio Paths")

    simulated = all_simulations[chosen]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(simulated[:, :500], alpha=0.3)
    ax.set_title("Simulated Portfolio Paths (Cholesky)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative Value")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    final_cum = simulated[-1, :] - 1
    mean_r = np.mean(final_cum)
    median_r = np.median(final_cum)
    q1 = np.percentile(final_cum, 25)
    q3 = np.percentile(final_cum, 75)
    prob_win = np.mean(final_cum > 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Cumulative Return", f"{mean_r:.2%}")
    col2.metric("Median Cumulative Return", f"{median_r:.2%}")
    col3.metric("P(Positive Return)", f"{prob_win:.2%}")

    col4, col5 = st.columns(2)
    col4.metric("1st Quartile (25th)", f"{q1:.2%}")
    col5.metric("3rd Quartile (75th)", f"{q3:.2%}")

    st.write(f"Simulation horizon: **{int(n_days)} days**")
