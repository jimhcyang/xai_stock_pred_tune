"""
create_daily_X_y.py

Fetches daily OHLC data for each ticker (2013-01-01 to 2025-01-10),
computes technical indicators, merges with shared (market + FRED) data,
and finally trims to 2015-01-01 through 2024-12-31 after adding the future target `y`.
All data is forward-filled and reindexed to business days.

Output:
    - CSV per ticker in data/daily_data/
    - Combined dictionary pickle: temp_output/daily_X_y.pkl

Author: Jim Yang
Date: 2025-04-05
"""

import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

# --------------------------------------------------------------------------------
# CONFIG / GLOBALS
# --------------------------------------------------------------------------------
FRED_API_KEY = "9cf8d916c926c5c0cad06d2c967cfccf"

# Your main directories
daily_data_dir = os.path.join("data", "daily_data")
os.makedirs(daily_data_dir, exist_ok=True)

output_dir = "temp_output"
os.makedirs(output_dir, exist_ok=True)

# Example Tickers
sector_tickers = [
    "AAPL", "MSFT",       # Info Tech
    "LLY", "UNH",         # Health Care
    "V", "MA",            # Financials
    "GOOGL", "META",      # Communication Services
    "AMZN", "TSLA",       # Consumer Discretionary
    "PG", "WMT",          # Consumer Staples
    "RTX", "UNP",         # Industrials
    "XOM", "CVX",         # Energy
    "LIN", "SHW",         # Materials
    "AMT", "PLD",         # Real Estate
    "NEE", "SO",          # Utilities
    "^GSPC"               # S&P 500
]

market_tickers = [
    "^GSPC", "^IXIC", "^DJI", "^VIX",
    "CL=F", "GC=F", "SI=F",
    "^TNX", "DX-Y.NYB"
]

fred_map = { "EFFR": "EFFR" }  # Example: Effective Fed Funds Rate

rename_map = {
    "SMA(5)":         "SMA5",
    "SMA(50)":        "SMA50",
    "SMA(200)":       "SMA200",
    "MACD_line":      "MACDLine",
    "MACD_signal":    "MACDSignal",
    "MACD_hist":      "MACDHist",
    "RSI(14)":        "RSI14",
    "BB_up(20,2)":    "BBupper",
    "BB_low(20,2)":   "BBlower",
    "ROC(12)":        "ROC12",
    "Stoch_K(14,3)":  "StochK",
    "Stoch_D(14,3)":  "StochD",
    "WilliamsR(14)":  "WillR",
    "AccDist":        "AccDist",
    "+DI(14)":        "PlusDI14",
    "PPO(5,10)":      "PPO",
    "TRANGE":         "TR",
    "MOM(5)":         "MOM5",
    "SlowStochD(5,10)": "SlowStochD",
    "ChaikinOsc(3,10)": "ChaikinOsc",
    "ADX(14)":        "ADX14",
    "ATR(14)":        "ATR14",
    "EFFR":           "FedFundsRate"
}

# --------------------------------------------------------------------------------
# FETCH FUNCTIONS (with business-day filtering)
# --------------------------------------------------------------------------------
def fetch_daily_ohlc_data(ticker, start_date="2013-01-01", end_date="2025-01-10"):
    """
    Fetch daily OHLC + Adj Close + Volume for a single ticker from Yahoo.
    Reindexes to business days and forward fills. Logs with [INFO]/[ERROR].
    """
    print(f"[INFO] Fetching daily OHLC for {ticker} ...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if df.empty:
            print(f"[ERROR] No data found for {ticker}")
            return pd.DataFrame()

        # Flatten if multi-level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = None
        df.columns.name = None
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch {ticker}: {e}")
        return pd.DataFrame()

def fetch_daily_adj_closed_data(tickers, start_date="2013-01-01", end_date="2025-01-10"):
    """
    Fetch daily Adj Close for multiple tickers at once.
    Reindexes to business days and forward fills. Logs with [INFO]/[ERROR].
    """
    print(f"[INFO] Fetching daily Adj Close for {tickers} ...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date,
                           group_by='ticker', auto_adjust=False, progress=False)
        if data.empty:
            print("[ERROR] No multi-ticker data found.")
            return pd.DataFrame()

        # Build the DataFrame
        adj_close_df = pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            # data.columns.levels[0] are ticker symbols
            ticker_levels = data.columns.levels[0]
            for tkr in tickers:
                if tkr in ticker_levels:
                    adj_close_df[tkr] = data[tkr]["Adj Close"]
        else:
            # Single-level columns fallback
            if "Adj Close" in data.columns:
                adj_close_df = data["Adj Close"].to_frame(name=tickers[0])

        # Reindex to business days
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        adj_close_df = adj_close_df.reindex(date_range).ffill()

        adj_close_df.index.name = None
        adj_close_df.columns.name = None
        return adj_close_df
    except Exception as e:
        print(f"[ERROR] Failed to fetch multi-ticker Adj Close: {e}")
        return pd.DataFrame()

def fetch_fred_data(fred_series_map, start_date="2013-01-01", end_date="2025-01-10", api_key=None):
    """
    Fetch daily or available frequency from FRED, reindexed to business days.
    Logs with [INFO]/[ERROR].
    """
    if not fred_series_map:
        return pd.DataFrame()
    print(f"[INFO] Fetching FRED data: {fred_series_map.keys()} ...")
    try:
        series_ids = list(fred_series_map.values())
        df_fred = web.DataReader(series_ids, "fred", start_date, end_date, api_key=api_key)
        if isinstance(df_fred, pd.Series):
            df_fred = df_fred.to_frame()

        # Rename columns to friendly names
        rename_dict = {}
        for friendly, fred_id in fred_series_map.items():
            if fred_id in df_fred.columns:
                rename_dict[fred_id] = friendly
        df_fred = df_fred.rename(columns=rename_dict)

        # Reindex to business days
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        df_fred = df_fred.reindex(date_range).ffill()
        return df_fred
    except Exception as e:
        print(f"[ERROR] FRED fetch failed: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------
# TECHNICAL INDICATORS (Identical to your existing definitions)
# --------------------------------------------------------------------------------
def SMA(series: pd.Series, window=50) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def MACD(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def RSI(series: pd.Series, window=14) -> pd.Series:
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def BBANDS(series: pd.Series, window=20, num_std=2):
    mid = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    upper = mid + (num_std * rolling_std)
    lower = mid - (num_std * rolling_std)
    return mid, upper, lower

def ROC(series: pd.Series, window=12) -> pd.Series:
    shifted = series.shift(window)
    diff = series - shifted
    return (diff / shifted) * 100

def STOCH_KD(df: pd.DataFrame, k_window=14, d_window=3):
    high = df['High']
    low  = df['Low']
    close= df['Close']
    hh = high.rolling(k_window, min_periods=k_window).max()
    ll = low.rolling(k_window, min_periods=k_window).min()
    stoch_k = 100 * (close - ll) / (hh - ll)
    stoch_d = stoch_k.rolling(d_window, min_periods=d_window).mean()
    return stoch_k, stoch_d

def WILLIAMS_R(df: pd.DataFrame, period=14):
    high = df['High']
    low  = df['Low']
    close= df['Close']
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return -100 * ((hh - close) / (hh - ll))

def ACCUM_DIST(df: pd.DataFrame):
    high = df['High']
    low  = df['Low']
    close= df['Close']
    vol  = df['Volume']
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm * vol
    return mfv.cumsum()

def PLUS_DI(df: pd.DataFrame, window=14):
    high = df['High']
    low  = df['Low']
    close= df['Close']
    up_move   = high.diff()
    down_move = -1 * low.diff()
    plus_dm   = np.where((up_move > 0) & (up_move > down_move), up_move, 0.0)
    prev_close= close.shift(1)
    r1 = high - low
    r2 = (high - prev_close).abs()
    r3 = (low - prev_close).abs()
    true_range = pd.concat([r1, r2, r3], axis=1).max(axis=1)
    plus_dm_roll = pd.Series(plus_dm, index=df.index).rolling(window).sum()
    tr_roll      = true_range.rolling(window).sum()
    return 100 * (plus_dm_roll / tr_roll)

def PPO(series: pd.Series, fast=5, slow=10):
    fast_ema = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return 100 * (fast_ema - slow_ema) / slow_ema

def TRANGE(df: pd.DataFrame):
    high = df['High']
    low  = df['Low']
    close= df['Close']
    prev_close = close.shift(1)
    r1 = high - low
    r2 = (high - prev_close).abs()
    r3 = (low - prev_close).abs()
    return pd.concat([r1, r2, r3], axis=1).max(axis=1)

def MOM(series: pd.Series, window=5):
    return series.diff(window)

def SD_stoch(df: pd.DataFrame, k_window=5, d_window=10):
    high = df['High']
    low  = df['Low']
    close= df['Close']
    hh = high.rolling(k_window, min_periods=k_window).max()
    ll = low.rolling(k_window, min_periods=k_window).min()
    raw_k = 100 * (close - ll) / (hh - ll)
    smooth_k = raw_k.rolling(d_window, min_periods=d_window).mean()
    return smooth_k.rolling(d_window, min_periods=d_window).mean()

def CO_chaikin(df: pd.DataFrame, short=3, long=10):
    ad_line = ACCUM_DIST(df)
    ad_ema_short = ad_line.ewm(span=short, adjust=False).mean()
    ad_ema_long  = ad_line.ewm(span=long,  adjust=False).mean()
    return ad_ema_short - ad_ema_long

def ADX(df: pd.DataFrame, window=14):
    high = df['High']
    low  = df['Low']
    close= df['Close']
    up_move   = high.diff()
    down_move = -1 * low.diff()
    plus_dm   = np.where((up_move > 0) & (up_move > down_move), up_move, 0.0)
    minus_dm  = np.where((down_move > 0) & (down_move > up_move), down_move, 0.0)
    prev_close= close.shift(1)
    r1 = high - low
    r2 = (high - prev_close).abs()
    r3 = (low - prev_close).abs()
    true_range = pd.concat([r1, r2, r3], axis=1).max(axis=1)
    plus_dm_roll  = pd.Series(plus_dm, index=df.index).rolling(window).sum()
    minus_dm_roll = pd.Series(minus_dm, index=df.index).rolling(window).sum()
    tr_roll       = true_range.rolling(window).sum()
    plus_di  = 100 * (plus_dm_roll / tr_roll)
    minus_di = 100 * (minus_dm_roll / tr_roll)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window).mean()

def ATR(df: pd.DataFrame, window=14):
    high = df['High']
    low  = df['Low']
    close= df['Close']
    prev_close = close.shift(1)
    r1 = high - low
    r2 = (high - prev_close).abs()
    r3 = (low - prev_close).abs()
    tr = pd.concat([r1, r2, r3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a broad set of indicators, using:
      - 'Adj Close' for trend/momentum-based
      - 'Close','High','Low','Volume' for range-based
    """
    if 'Adj Close' not in df.columns:
        raise ValueError("[ERROR] DataFrame must include 'Adj Close' for compute_all_indicators.")

    if not {'Close','High','Low','Volume'}.issubset(df.columns):
        raise ValueError("[ERROR] DataFrame missing one of {'Close','High','Low','Volume'}")

    out = pd.DataFrame(index=df.index)

    c_adj = df['Adj Close']
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']

    # Price trend
    out['SMA(5)']   = SMA(c_adj, 5)
    out['SMA(50)']  = SMA(c_adj, 50)
    out['SMA(200)'] = SMA(c_adj, 200)

    macd_line, macd_signal, macd_hist = MACD(c_adj, 12, 26, 9)
    out['MACD_line']   = macd_line
    out['MACD_signal'] = macd_signal
    out['MACD_hist']   = macd_hist

    out['RSI(14)']  = RSI(c_adj, 14)
    _, bb_up, bb_low = BBANDS(c_adj, 20, 2)
    out['BB_up(20,2)']  = bb_up
    out['BB_low(20,2)'] = bb_low

    out['ROC(12)']  = ROC(c_adj, 12)
    out['PPO(5,10)'] = PPO(c_adj, 5, 10)
    out['MOM(5)']    = MOM(c_adj, 5)

    # Range/volume
    ohlcv = pd.DataFrame({'High': h, 'Low': l, 'Close': c, 'Volume': v}, index=df.index)
    stoch_k, stoch_d = STOCH_KD(ohlcv, 14, 3)
    out['Stoch_K(14,3)'] = stoch_k
    out['Stoch_D(14,3)'] = stoch_d

    out['WilliamsR(14)'] = WILLIAMS_R(ohlcv, 14)
    out['AccDist']       = ACCUM_DIST(ohlcv)
    out['+DI(14)']       = PLUS_DI(ohlcv, 14)
    out['TRANGE']        = TRANGE(ohlcv)
    out['SlowStochD(5,10)'] = SD_stoch(ohlcv, 5, 10)
    out['ChaikinOsc(3,10)'] = CO_chaikin(ohlcv, 3, 10)
    out['ADX(14)']          = ADX(ohlcv, 14)
    out['ATR(14)']          = ATR(ohlcv, 14)

    return out

# --------------------------------------------------------------------------------
# CLEANING FUNCTION
# --------------------------------------------------------------------------------
def clean_daily_dataframe(
    df: pd.DataFrame,
    target_col: str = "Adj Close",
    forecast_horizon: int = 1,
    raw_start_date: str = "2013-01-01",
    raw_end_date: str = "2025-01-10",
    final_start_date: str = "2015-01-01",
    final_end_date: str = "2024-12-31",
    fillna: bool = True
) -> pd.DataFrame:
    """
    1) Data from [raw_start_date, raw_end_date].
    2) Shift target_col by forecast_horizon => column 'y'.
    3) Drop NaN 'y' rows.
    4) Trim to [final_start_date, final_end_date].
    5) Forward-fill if fillna=True.
    """
    if target_col not in df.columns:
        raise KeyError(f"[ERROR] Target column '{target_col}' not found in DataFrame.")

    out = df.copy()
    out['y'] = out[target_col].shift(-forecast_horizon)
    out.dropna(subset=['y'], inplace=True)

    # final range
    mask = (out.index >= pd.to_datetime(final_start_date)) & (out.index <= pd.to_datetime(final_end_date))
    out = out.loc[mask]

    if fillna:
        out = out.ffill()

    return out

# --------------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------------
def run_daily_pipeline_for_all_tickers(
    tickers,
    start_date="2013-01-01",
    end_date="2025-01-10",
    forecast_horizon=1,
    market_tickers=None,
    fred_series_map=None,
    rename_map=None,
    fred_api_key=None,
    save_dir="daily_data"
):
    """
    1) Pre-fetch market + FRED => shared_df
    2) For each ticker:
       - fetch its OHLC
       - compute indicators
       - join with shared_df
       - rename columns
       - clean_daily_dataframe
       - save CSV
    3) Return dict of final DataFrames
    """
    if market_tickers is None:
        market_tickers = []
    if fred_series_map is None:
        fred_series_map = {}

    os.makedirs(save_dir, exist_ok=True)
    daily_X_y = {}

    print("[INFO] Fetching shared data (market + FRED)...")
    df_market = fetch_daily_adj_closed_data(market_tickers, start_date, end_date)
    df_fred   = fetch_fred_data(fred_series_map, start_date, end_date, api_key=fred_api_key)
    shared_df = df_market.join(df_fred, how='left').sort_index()

    for ticker in tickers:
        print(f"[INFO] Processing {ticker}...")
        safe_ticker = ticker.replace('^','')
        csv_filename = f"{safe_ticker}_daily.csv"

        try:
            # 1) Single ticker OHLC
            df_stock = fetch_daily_ohlc_data(ticker, start_date, end_date)
            if df_stock.empty:
                print(f"[ERROR] No data for {ticker}, skipping...")
                continue

            # 2) Compute technical indicators
            df_tech = compute_all_indicators(df_stock)

            # 3) Merge them: Full single-ticker set
            df_ticker_full = df_stock.join(df_tech, how='left')

            # 4) Merge in the shared data (other tickers, FRED)
            df_merged = df_ticker_full.join(shared_df, how='left')

            # 5) Rename columns (optional)
            if rename_map:
                df_merged = df_merged.rename(columns=rename_map)

            # 6) Final cleaning: add `y` and trim
            final_df = clean_daily_dataframe(
                df=df_merged,
                target_col="Adj Close",  # or "Close" if you prefer
                forecast_horizon=forecast_horizon,
                raw_start_date=start_date,
                raw_end_date=end_date,
                final_start_date="2015-01-01",
                final_end_date="2024-12-31",
                fillna=True
            )

            # 7) Save CSV
            out_path = os.path.join(save_dir, csv_filename)
            final_df.to_csv(out_path)
            print(f"[SUCCESS] {ticker} -> {csv_filename}, shape={final_df.shape}")

            daily_X_y[ticker] = final_df
        except Exception as e:
            print(f"[ERROR] {ticker} pipeline failed: {e}")

    return daily_X_y

# --------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Running daily pipeline for all tickers...")
    daily_X_y = run_daily_pipeline_for_all_tickers(
        tickers=sector_tickers,
        start_date="2013-01-01",
        end_date="2025-01-10",
        forecast_horizon=1,  # or 5, as you wish
        market_tickers=market_tickers,
        fred_series_map=fred_map,
        rename_map=rename_map,
        fred_api_key=FRED_API_KEY,
        save_dir=daily_data_dir
    )

    # Save the final dictionary
    pickle_path = os.path.join(output_dir, "daily_X_y.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(daily_X_y, f)

    print(f"[SUCCESS] Final daily_X_y saved to {pickle_path}")
