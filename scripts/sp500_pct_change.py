#!/usr/bin/env python3
"""
Fetch S&P 500 historical data and plot year-over-year percent changes since a start year.

Usage examples:
  python3 scripts/sp500_pct_change.py --start-year 2008 --save sp500_yoy.png --no-show

This script uses yfinance to fetch '^GSPC' data by default.
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def fetch_data(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """Download daily OHLC data for `ticker` using yfinance.

    Returns a DataFrame with DatetimeIndex and columns including 'Close'.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required to fetch data. Install with `pip install yfinance`"
        ) from e

    # yfinance expects strings like '2007-01-01'
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker} from {start_date} to {end_date}")
    return df


def compute_yearly_last_close(df: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by year (int) of the last closing price in each calendar year."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Resample by calendar year end and take last available close.
    # Newer pandas versions reject the alias 'Y' and expect 'YE' for year-end.
    # Try 'Y' first for backwards compatibility, fall back to 'YE' if needed.
    last_exc = None
    for freq in ('Y', 'YE'):
        try:
            yearly = df['Close'].resample(freq).last()
            break
        except Exception as e:  # catch ValueError or pandas-specific errors
            last_exc = e
            continue
    else:
        # re-raise the last exception with context
        raise last_exc
    # Convert index to year integer
    yearly.index = yearly.index.year
    return yearly


def compute_year_over_year_pct(yearly_close: pd.Series) -> pd.DataFrame:
    """Given yearly_close indexed by year, compute percent change YoY.

    Returns DataFrame with columns: Year, Close, PctChange
    """
    # Be robust if a DataFrame or Series is passed in.
    if isinstance(yearly_close, pd.DataFrame):
        # prefer a 'Close' column if present, otherwise take the first column
        if 'Close' in yearly_close.columns:
            s = yearly_close['Close'].squeeze()
        else:
            s = yearly_close.iloc[:, 0].squeeze()
    else:
        s = yearly_close.squeeze()

    # Ensure we have a Series named 'Close'
    s = pd.Series(s, name='Close')

    df = s.to_frame().reset_index()
    # Ensure the first column is named 'Year'
    first_col = df.columns[0]
    if first_col != 'Year':
        df = df.rename(columns={first_col: 'Year'})

    df['PctChange'] = df['Close'].pct_change() * 100.0
    df = df.sort_values('Year')
    return df


def plot_bar(df: pd.DataFrame, start_year: int, save_path: str = None, show: bool = True):
    """Plot a bar chart of percent changes.

    df must contain columns 'Year' and 'PctChange'.
    """
    plot_df = df[df['Year'] >= start_year].dropna(subset=['PctChange'])
    if plot_df.empty:
        raise RuntimeError(f"No data to plot for years >= {start_year}")

    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))

    # Choose a diverging palette so negative years show distinct color
    palette = sns.diverging_palette(10, 150, n=len(plot_df))

    bars = sns.barplot(x='Year', y='PctChange', data=plot_df, palette=palette)
    bars.set_title(f"S&P 500 Year-over-Year Percent Change (since {start_year})")
    bars.set_ylabel('Percent change (%)')
    bars.set_xlabel('Year')

    # Annotate bars with values
    for p in bars.patches:
        height = p.get_height()
        if pd.isna(height):
            continue
        bars.annotate(f"{height:.1f}%",
                      (p.get_x() + p.get_width() / 2., height),
                      ha='center', va='bottom' if height >= 0 else 'top',
                      fontsize=9, color='black', xytext=(0, 3 if height >= 0 else -3),
                      textcoords='offset points')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved bar chart to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def parse_args():
    p = argparse.ArgumentParser(description='Plot S&P 500 year-over-year percent changes')
    p.add_argument('--start-year', type=int, default=2008, help='Start year for plotting (inclusive)')
    p.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol for S&P index (default: ^GSPC)')
    p.add_argument('--save', type=str, default=None, help='Path to save the plotted PNG (optional)')
    p.add_argument('--no-show', action='store_true', help='Do not display the plot interactively')
    p.add_argument('--cache-csv', type=str, default=None, help='Optional CSV path to cache downloaded data')
    return p.parse_args()


def main():
    args = parse_args()
    start_year = args.start_year

    # We need data from year-before-start to compute the first YoY change
    data_start = f"{start_year - 1}-01-01"
    data_end = None  # let yfinance fetch up to today

    # Optionally load from cache
    if args.cache_csv and os.path.exists(args.cache_csv):
        print(f"Loading cached data from {args.cache_csv}")
        df = pd.read_csv(args.cache_csv, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {args.ticker} data starting {data_start} ...")
        df = fetch_data(args.ticker, data_start, data_end)
        if args.cache_csv:
            os.makedirs(os.path.dirname(args.cache_csv), exist_ok=True)
            df.to_csv(args.cache_csv)
            print(f"Cached fetched data to {args.cache_csv}")

    yearly = compute_yearly_last_close(df)
    yoy = compute_year_over_year_pct(yearly)

    plot_bar(yoy, start_year=start_year, save_path=args.save, show=(not args.no_show))


if __name__ == '__main__':
    main()
