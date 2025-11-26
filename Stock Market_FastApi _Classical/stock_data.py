import yfinance as yf
from datetime import timedelta

def get_stock_history_data(date, config):
    start_date = date - timedelta(days=config['days'])
    stock_data = yf.download(config['stock_symbol'], start=start_date, end=date)
    return stock_data