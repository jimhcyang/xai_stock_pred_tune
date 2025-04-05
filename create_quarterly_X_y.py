"""
create_quarterly_X_y.py

Combines:
1) Retrieval of quarterly financial statements from QuickFS
2) Processing and validation of those statements
3) Retrieval of quarterly stock data and economic indicators
4) Merging all data into a final dataset (X, y)
5) Saving the combined outputs to a pickle file

Comments about each ticker and each financial/economic indicator are retained.
Comments about internal code processes have been streamlined or removed.
All messages indicate [SUCCESS] or [ERROR] for clarity.

Author: Jim Yang
Date: 2025-04-05
"""

import os
import glob
import pickle
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime
from dateutil.relativedelta import relativedelta


# --------------------------------------------------------------------------------
# QuickFS Setup
# --------------------------------------------------------------------------------
QUICKFS_API_KEY = "49043bec203bf8503262185a94427de03091c7d5"

# Directory to store statements
statements_dir = os.path.join("data", "statements")
os.makedirs(statements_dir, exist_ok=True)

# Base URL for QuickFS
base_url = "https://public-api.quickfs.net/v1/data/all-data/"

# Headers (QuickFS requires an API key in the header)
headers = {"X-QFS-API-Key": QUICKFS_API_KEY}

# --------------------------------------------------------------------------------
# Tickers of Interest
# --------------------------------------------------------------------------------
# Explanation: 
# These tickers cover major sectors of the US market. The inline comments describe
# which sector they represent.
sector_tickers = [
    "AAPL", "MSFT",      # Info Tech
    "LLY", "UNH",        # Health Care
    "V", "MA",           # Financials
    "GOOGL", "META",     # Communication Services
    "AMZN", "TSLA",      # Consumer Discretionary
    "PG", "WMT",         # Consumer Staples
    "RTX", "UNP",        # Industrials
    "XOM", "CVX",        # Energy
    "LIN", "SHW",        # Materials
    "AMT", "PLD",        # Real Estate
    "NEE", "SO",         # Utilities
]

# --------------------------------------------------------------------------------
# Fetch Quarterly Financial Statements from QuickFS
# --------------------------------------------------------------------------------
def fetch_quickfs_statements(tickers, save_dir, base_url, headers):
    """
    Fetches quarterly statements from QuickFS for each ticker and saves as CSV.
    Prints [SUCCESS] or [ERROR] messages accordingly.
    """
    for ticker in tickers:
        qfs_symbol = f"{ticker}:US"
        safe_ticker = ticker.replace("-", "_")  # For file naming
        
        print(f"[INFO] Fetching {qfs_symbol}...")
        try:
            response = requests.get(f"{base_url}{qfs_symbol}", headers=headers)
            response.raise_for_status()
            data = response.json()

            # Extract the quarterly data portion
            quarterly_data = data["data"]["financials"]["quarterly"]
            periods = quarterly_data["period_end_date"]

            # Create a DataFrame with metrics as rows and periods as columns
            records = {key: quarterly_data[key] for key in quarterly_data if key != "period_end_date"}
            df = pd.DataFrame(records, index=periods).T

            # Save to CSV
            save_path = os.path.join(save_dir, f"{safe_ticker}.csv")
            df.to_csv(save_path)
            print(f"[SUCCESS] Saved {ticker} statements to {save_path}")

        except Exception as e:
            print(f"[ERROR] Failed to fetch {ticker} from QuickFS: {e}")

# --------------------------------------------------------------------------------
# Process and Validate Statements
# --------------------------------------------------------------------------------
# Explanation: Each variable below is a fundamental company metric, with a brief
# description on why it's important for fundamental analysis.
selected_columns = [
    "revenue",                    # Top-line; shows how much money is coming in
    "gross_profit",               # Core profitability; after cost of goods sold
    "operating_income",           # Income from core operations; pre-interest/taxes
    "net_income",                 # Bottom-line performance; ultimate profitability
    "eps_diluted",                # Normalized earnings per share
    "fcf",                        # Free cash flow
    "cf_cfo",                     # Total cash from operations
    "capex",                      # Capital investments
    "total_assets",               # Measures company size
    "total_liabilities",          # Obligations; leverage and risk
    "total_equity",               # Net worth; base for ROE/solvency
    "debt_to_equity",             # Capital structure risk vs growth
    "cash_and_equiv",             # Liquidity buffer
    "working_capital",            # Short-term financial health
    "roa",                        # Return on assets; efficiency metric
    "roe",                        # Return on equity; profitability for shareholders
    "operating_margin",           # Revenue converted to operating profit
    "receivables",                # Amount owed by customers
    "inventories",                # Production/sales balance
    "ppe_net",                    # Physical asset base
    "cff_debt_net",               # Net debt raised/paid
    "cogs",                       # Cost of goods sold
    "sga",                        # Overhead costs
    "revenue_growth",             # Growth trajectory
]

def process_financial_statements(directory):
    """
    Ensures each CSV in 'directory' has the required rows/period coverage.
    Prints [SUCCESS] or [ERROR] messages accordingly.
    """
    expected_shape = (48, len(selected_columns))  # 48 quarters x number of columns
    start_period = "2013Q1"
    end_period   = "2024Q4"

    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        print(f"[INFO] No CSV files found in {directory}")
        return

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        print(f"[INFO] Validating {filename}...")
        try:
            df = pd.read_csv(csv_path)
            cols = df.columns.tolist()
            if not cols:
                print(f"[ERROR] File {filename} has no columns")
                continue

            # Rename first column to blank, use it as index
            cols[0] = ""
            df.columns = cols
            df.set_index(cols[0], inplace=True)

            try:
                df.columns = pd.PeriodIndex(df.columns, freq='Q')
            except Exception as e:
                print(f"[ERROR] Could not convert columns to PeriodIndex in {filename}: {e}")
                continue

            # Ensure all selected_columns exist
            missing_cols = [col for col in selected_columns if col not in df.index]
            if missing_cols:
                print(f"[ERROR] Missing rows in {filename}: {missing_cols}")
                continue

            # Subset data
            sub_df = df.loc[selected_columns, start_period:end_period].T

            # Check final shape
            if sub_df.shape != expected_shape:
                print(f"[ERROR] Shape mismatch in {filename}: got {sub_df.shape}, expected {expected_shape}")
                continue

            print(f"[SUCCESS] {filename} validated successfully.")

        except Exception as e:
            print(f"[ERROR] Failed to process {filename} due to an unexpected exception: {e}")

# --------------------------------------------------------------------------------
# Economic Data (FRED/BLS)
# --------------------------------------------------------------------------------
# Explanation: Each item in these dictionaries is an important macroeconomic indicator
# that can affect company performance in different ways.
fred_series = {
    "GDP": "GDP",                            # Gross Domestic Product
    "Real_GDP": "GDPC1",                     # Real Gross Domestic Product
    "Real_GDP_Per_Capita": "A939RX0Q048SBEA",# Real GDP Per Capita
    "GDP_Deflator": "GDPDEF",                # GDP Deflator (Inflation)
    "Unemployment": "UNRATE",                # Unemployment Rate
    "M2_Money_Supply": "M2SL",               # M2 Money Supply
    "CPI": "CPIAUCSL",                       # Consumer Price Index
    "Fed_Funds_Rate": "FEDFUNDS",            # Federal Funds Rate
    "Consumer_Sentiment": "UMCSENT",         # Consumer Sentiment
    "Retail_Sales": "RSAFS",                 # Retail Sales
    "Industrial_Production": "INDPRO",       # Industrial Production
    "Housing_Starts": "HOUST",               # Housing Starts
    "Corp_Profits": "CP",                    # Corporate Profits
    "PCE": "PCE",                            # Personal Consumption Expenditures
    "Business_Investment": "PNFI"            # Private Nonresidential Fixed Investment
}

bls_series = {
    "Labor_Force_Participation": "LNS11300000",            # Labor Force Participation Rate
    "Job_Openings_Rate": "JTS000000000000000JOR",          # Job Openings Rate
    "Real_Weekly_Earnings": "CES0500000032",               # Real Average Weekly Earnings
    "Private_Job_Growth": "CES0500000001",                 # Total Private Job Growth
    "CPI_BLS": "CUSR0000SA0",                              # Consumer Price Index (BLS)
    "Hourly_Earnings": "CES0500000003",                    # Average Hourly Earnings
}

FRED_API_KEY = "9cf8d916c926c5c0cad06d2c967cfccf"
BLS_API_KEY = "dbdf67bc2b0b4a909b07543464457938"

def fetch_quarterly_stock_data(ticker, start_date, end_date):
    """
    Fetch quarterly data (Close, High, Low) for a given stock ticker.
    Returns a DataFrame indexed by Quarter.
    """
    try:
        daily_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if daily_data.empty:
            print(f"[ERROR] No data found for {ticker}")
            return pd.DataFrame()

        quarterly_data = pd.DataFrame()
        quarterly_data['Close'] = daily_data['Adj Close'].resample('QE').last()
        quarterly_data['High'] = daily_data['High'].resample('QE').max()
        quarterly_data['Low'] = daily_data['Low'].resample('QE').min()
        quarterly_data['Quarter'] = quarterly_data.index.to_period('Q')
        quarterly_data.set_index('Quarter', inplace=True)

        return quarterly_data
    
    except Exception as e:
        print(f"[ERROR] Unable to fetch stock data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_quarterly_fred_data(series_dict, start_date, end_date, api_key=None):
    """
    Fetch quarterly economic data from FRED. 
    Returns a DataFrame with each series as a column, indexed by Quarter.
    """
    if not series_dict:
        return pd.DataFrame()
    try:
        series_ids = list(series_dict.values())
        df = web.DataReader(series_ids, "fred", start_date, end_date, api_key=api_key)
        reverse_map = {v: k for k, v in series_dict.items()}
        df = df.rename(columns=reverse_map)
        df_quarterly = df.resample('QE').last()
        df_quarterly['Quarter'] = df_quarterly.index.to_period('Q')
        df_quarterly.set_index('Quarter', inplace=True)
        return df_quarterly
    except Exception as e:
        print(f"[ERROR] Unable to fetch FRED data: {e}")
        return pd.DataFrame()

def fetch_bls_data(series_dict, start_year, end_year, api_key=None):
    """
    Fetch data from BLS in quarterly format. 
    Returns a DataFrame with each series as a column, indexed by Quarter.
    """
    if not series_dict or not api_key:
        return pd.DataFrame()
    
    date_range = pd.period_range(start=f"{start_year}Q1", end=f"{end_year}Q4", freq='Q')
    result_df = pd.DataFrame(index=date_range)
    series_ids = list(series_dict.values())
    
    for i in range(0, len(series_ids), 50):
        batch_series = series_ids[i:i+50]
        headers = {'Content-type': 'application/json'}
        payload = {
            "seriesid": batch_series,
            "startyear": str(start_year),
            "endyear": str(end_year),
            "registrationkey": api_key
        }
        try:
            url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            json_data = response.json()
            if json_data.get('status') != 'REQUEST_SUCCEEDED':
                print(f"[ERROR] BLS API error: {json_data.get('message', 'Unknown error')}")
                continue
            for series_obj in json_data.get('Results', {}).get('series', []):
                sid = series_obj.get('seriesID')
                if not sid:
                    continue
                var_name = None
                for k, v in series_dict.items():
                    if v == sid:
                        var_name = k
                        break
                if not var_name:
                    continue
                series_data = {}
                for item in series_obj.get('data', []):
                    if item.get('period') == 'M13':  # Skip annual
                        continue
                    try:
                        year = item.get('year')
                        month = int(item.get('period', 'M0')[1:])
                        quarter = (month - 1) // 3 + 1
                        period_str = f"{year}Q{quarter}"
                        period = pd.Period(period_str, freq='Q')
                        value = float(item.get('value', 0))
                        series_data[period] = value
                    except (ValueError, TypeError):
                        continue
                if series_data:
                    period_series = pd.Series(series_data)
                    result_df[var_name] = period_series
        except Exception as e:
            print(f"[ERROR] Unable to fetch BLS data: {e}")
    
    result_df.sort_index(inplace=True)
    return result_df

def fetch_economic_data(fred_series_map, bls_series_map, start_date, end_date,
                        fred_api_key=None, bls_api_key=None):
    """
    Fetches and merges FRED + BLS data into a single quarterly DataFrame.
    """
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year

    print("[INFO] Fetching FRED data...")
    fred_data = fetch_quarterly_fred_data(fred_series_map, start_date, end_date, api_key=fred_api_key)

    print("[INFO] Fetching BLS data...")
    bls_data = fetch_bls_data(bls_series_map, start_year, end_year, api_key=bls_api_key)

    econ_df = pd.concat([fred_data, bls_data], axis=1)
    return econ_df

def run_pipeline_for_ticker(ticker, start_date, end_date, economic_data=None,
                            save_dir="quarterly_data", csv_filename=None):
    """
    For a single ticker: fetch quarterly stock data, merge with economic data, and save to CSV.
    Prints [SUCCESS] or [ERROR].
    """
    print(f"[INFO] Processing {ticker}...")
    try:
        stock_data = fetch_quarterly_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            return None

        if economic_data is not None and not economic_data.empty:
            combined_data = stock_data.join(economic_data, how='left')
        else:
            combined_data = stock_data

        if csv_filename:
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, csv_filename)
            combined_data.to_csv(output_path)
            print(f"[SUCCESS] Saved combined data for {ticker} to {output_path}")

        return combined_data

    except Exception as e:
        print(f"[ERROR] {ticker} pipeline failed: {e}")
        return None

def run_quarterly_pipeline_for_all_tickers(tickers, fred_series_map=None, bls_series_map=None,
                                           start_date="2013-01-01", end_date="2024-12-31",
                                           fred_api_key=None, bls_api_key=None, save_dir="quarterly_data"):
    """
    Fetches economic data once. Then, for each ticker, runs the pipeline to get stock data
    and merges with economic data. Saves output CSV for each ticker. 
    Returns a dictionary of DataFrames for each ticker.
    """
    if fred_series_map is None:
        fred_series_map = {}
    if bls_series_map is None:
        bls_series_map = {}

    os.makedirs(save_dir, exist_ok=True)
    economic_data = fetch_economic_data(fred_series_map, bls_series_map,
                                        start_date, end_date, fred_api_key, bls_api_key)

    # Save the combined economic data
    if not economic_data.empty:
        econ_path = os.path.join(save_dir, "economic_data_quarterly.csv")
        economic_data.to_csv(econ_path)
        print(f"[SUCCESS] Saved economic data to {econ_path}")

    results = {}
    for ticker in tickers:
        file_out = f"{ticker.replace('^', '')}_quarterly.csv"
        df = run_pipeline_for_ticker(ticker, start_date, end_date,
                                     economic_data=economic_data,
                                     save_dir=save_dir,
                                     csv_filename=file_out)
        if df is not None:
            results[ticker] = df
    return results

# --------------------------------------------------------------------------------
# Merge Statements and Market/Econ Data
# --------------------------------------------------------------------------------
def load_and_merge_company(ticker, quarterly_dir, statements_dir):
    """
    1) Load the quarterly stock/econ CSV for this ticker.
    2) Load the QuickFS statements for this ticker.
    3) Merge them into a single DataFrame.
    Prints [SUCCESS] or [ERROR].
    """
    expected_shape = (48, len(selected_columns))  # 2013Q1 - 2024Q4
    start_period = pd.Period("2013Q1", freq="Q")
    end_period = pd.Period("2024Q4", freq="Q")

    quarterly_df_path = os.path.join(quarterly_dir, f"{ticker}_quarterly.csv")
    safe_ticker = ticker.replace("-", "_")
    company_path = os.path.join(statements_dir, f"{safe_ticker}.csv")

    if not os.path.exists(quarterly_df_path):
        print(f"[ERROR] Quarterly market/econ data for {ticker} not found.")
        return None
    if not os.path.exists(company_path):
        print(f"[ERROR] Company financial statements for {ticker} not found.")
        return None

    try:
        # Load stock/econ data
        qdf = pd.read_csv(quarterly_df_path, index_col=0)
        qdf.index = pd.PeriodIndex(qdf.index, freq="Q")

        # Load company statements
        raw_df = pd.read_csv(company_path, index_col=0)
        company_df = raw_df.T
        company_df.index = pd.to_datetime(company_df.index).to_period("Q")
        company_df = company_df.loc[start_period:end_period]

        # Check selected columns
        missing_cols = [col for col in selected_columns if col not in company_df.columns]
        if missing_cols:
            print(f"[ERROR] {ticker} missing required financial rows: {missing_cols}")
            return None

        # Filter only the selected columns
        company_df = company_df[selected_columns]
        if company_df.shape != expected_shape:
            print(f"[ERROR] {ticker} statement data shape mismatch. Got {company_df.shape}, expected {expected_shape}.")
            return None

        # Merge
        if not company_df.index.equals(qdf.index):
            print(f"[INFO] Index mismatch for {ticker}; using inner join.")
            merged_df = pd.concat([qdf, company_df], axis=1, join="inner")
        else:
            merged_df = pd.concat([qdf, company_df], axis=1)

        print(f"[SUCCESS] Merged {ticker}. Final shape: {merged_df.shape}")
        return merged_df

    except Exception as e:
        print(f"[ERROR] Failed to load/merge {ticker}: {e}")
        return None

def clean_quarterly_dataframe(df, start_q="2015Q1", end_q="2024Q4",
                              target_col="Close", forecast_horizon=1,
                              fillna=False, index_as_string=True):
    """
    Cleans the merged DataFrame:
    1) Filters to [start_q, end_q].
    2) Optionally forward fills missing data.
    3) Adds future target column 'y' by shifting 'target_col' up by 'forecast_horizon' quarters.
    4) Drops rows with NaN in 'y'.
    5) Optionally converts PeriodIndex to string ("YYYYQ#").
    """
    if not isinstance(df.index, pd.PeriodIndex):
        df.index = pd.PeriodIndex(df.index, freq='Q')

    start_q = pd.Period(start_q, freq='Q')
    end_q = pd.Period(end_q, freq='Q')
    df = df[(df.index >= start_q) & (df.index <= end_q)]

    if fillna:
        df = df.ffill()

    if target_col not in df.columns:
        raise KeyError(f"[ERROR] Target column '{target_col}' not found in DataFrame.")

    df["y"] = df[target_col].shift(-forecast_horizon)
    df.dropna(subset=["y"], inplace=True)

    if index_as_string:
        df.index = df.index.astype(str)

    return df

# --------------------------------------------------------------------------------
# Main Execution Flow
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Fetch QuickFS data
    fetch_quickfs_statements(sector_tickers, statements_dir, base_url, headers)

    # 2) Process financial statements
    process_financial_statements(statements_dir)

    # 3) Fetch stock & economic data
    stock_prices_q_path = os.path.join("data", "quarterly_data")
    os.makedirs(stock_prices_q_path, exist_ok=True)

    print("[INFO] Running pipeline for all tickers (stock+econ data)...")
    run_quarterly_pipeline_for_all_tickers(
        tickers=sector_tickers,
        fred_series_map=fred_series,
        bls_series_map=bls_series,
        start_date="2013-01-01",
        end_date="2024-12-31",
        fred_api_key=FRED_API_KEY,
        bls_api_key=BLS_API_KEY,
        save_dir=stock_prices_q_path
    )

    # 4) Merge statements with market & econ data; create final X, y
    quarterly_X_y = {}
    for tkr in sector_tickers:
        merged_df = load_and_merge_company(tkr, quarterly_dir=stock_prices_q_path, statements_dir=statements_dir)
        if merged_df is not None:
            cleaned_df = clean_quarterly_dataframe(
                df=merged_df,
                start_q="2015Q1",
                end_q="2024Q4",
                target_col="Close",
                forecast_horizon=1,
                fillna=True
            )
            quarterly_X_y[tkr] = cleaned_df

    # 5) Save final dictionary
    output_dir = "temp_output"
    os.makedirs(output_dir, exist_ok=True)
    pickle_path = os.path.join(output_dir, "quarterly_X_y.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(quarterly_X_y, f)

    print(f"[SUCCESS] Final quarterly_X_y saved to {pickle_path}")
