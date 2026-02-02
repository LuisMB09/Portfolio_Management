import os
import logging
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.rcParams['figure.facecolor'] = 'lightgrey'

PRICE_LOOKBACK_YEARS = 1


def get_financials(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    company = yf.Ticker(ticker)
    bs = company.balance_sheet.iloc[:, :-1]
    ist = company.income_stmt.iloc[:, :-1]
    cf = company.cash_flow.iloc[:, :-1]
    return bs, ist, cf


def get_historical_price(ticker: str, start: datetime.date) -> float:
    end = start + datetime.timedelta(days=4)
    price = pd.DataFrame(yf.Ticker(ticker).history(start=start, end=end)["Close"])
    if price.empty:
        raise ValueError(f"No price data for {ticker} between {start} and {end}")
    return price.iloc[0, 0]


def compute_ratios(bs: pd.DataFrame, ist: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'Gross Margin': ist.loc['Gross Profit'] / ist.loc['Total Revenue'],
        'Operating Margin': ist.loc['Operating Income'] / ist.loc['Total Revenue'],
        'Net Margin': ist.loc['Net Income Common Stockholders'] / ist.loc['Total Revenue'],
        'ROE': ist.loc['Net Income Common Stockholders'] / bs.loc['Stockholders Equity'],
        'Revenue': ist.loc['Total Revenue'],
        'FCF to Sales': cf.loc['Free Cash Flow'] / ist.loc['Total Revenue'],
        'Current Ratio': bs.loc['Current Assets'] / bs.loc['Current Liabilities'],
        'Solvency': bs.loc['Total Assets'] / bs.loc['Total Liabilities Net Minority Interest'],
    }).sort_index()


def download_price_data(ticker: str) -> tuple[pd.Series, float]:
    lookback_start = datetime.date.today() - relativedelta(years=PRICE_LOOKBACK_YEARS)
    price = yf.download(ticker, start=lookback_start, progress=False, auto_adjust=True)['Close']
    daily_returns = price.pct_change().dropna()
    vol = (daily_returns.std() * np.sqrt(252)).values[0]
    return price, vol


def plot_overview(ratios: pd.DataFrame, price: pd.Series, vol: float, ticker: str, save_path: str) -> None:
    fig = plt.figure(figsize=(30, 14))
    gs = fig.add_gridspec(3, 4)

    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(price, color='navy')
    ax_price.set_title(f'{ticker} Stock Price | Volatility: {vol:.2%}')
    ax_price.set_ylabel('Price')
    ax_price.grid()

    axes = [fig.add_subplot(gs[r, c]) for r in range(1, 3) for c in range(4)]

    for ax, column in zip(axes, ratios.columns):
        ax.plot(ratios.index, ratios[column], color='navy', marker='o')
        ax.set_title(column)
        ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{ticker}_overview.png"), dpi=300, bbox_inches='tight')
    plt.close()


def process_ticker(ticker: str, base_dir: str) -> str:
    ticker_dir = os.path.join(base_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    try:
        bs, ist, cf = get_financials(ticker)
        ratios = compute_ratios(bs, ist, cf)
        price, vol = download_price_data(ticker)
        plot_overview(ratios, price, vol, ticker, ticker_dir)
        return f"Saved combined plot for {ticker}"
    except Exception as e:
        logger.exception(f"Failed to process {ticker}")
        return f"An error occurred for {ticker}: {e}"


def main():
    base_dir = "Financials"
    os.makedirs(base_dir, exist_ok=True)

    tickers = ['UBER']

    max_workers = min(len(tickers), os.cpu_count())
    logger.info(f"Using {max_workers} workers for parallel processing.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_ticker, t, base_dir) for t in tickers]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
