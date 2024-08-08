import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(ticker, csv_filename):
    if os.path.exists(csv_filename):
        print(f"Reading data from {csv_filename}")
        return pd.read_csv(csv_filename, parse_dates=['Date'])
    else:
        try:
            print(f"Fetching data from yfinance API for {ticker}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5 * 365)  # 5 years of data
            data = yf.download(ticker, start=start_date, end=end_date)
            data = data.reset_index()
            data.to_csv(csv_filename, index=False)
            return data
        except:
            print("Error occurred during data loading")

def prepare_stock_data(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.sort_values('Date')
    return stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]