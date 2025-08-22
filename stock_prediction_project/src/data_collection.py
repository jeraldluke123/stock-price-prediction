import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class StockDataCollector:
    """
    Collects stock data from Yahoo Finance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fetch_stock_data(self, symbol, period="2y", interval="1d"):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Period to fetch data for ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            self.logger.info(f"Fetching data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            self.logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def fetch_multiple_stocks(self, symbols, period="2y", interval="1d"):
        """
        Fetch data for multiple stock symbols
        
        Args:
            symbols (list): List of stock symbols
            period (str): Period to fetch data for
            interval (str): Data interval
        
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, period, interval)
                stock_data[symbol] = data
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return stock_data
    
    def get_stock_info(self, symbol):
        """
        Get additional stock information
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            self.logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    def validate_data(self, data):
        """
        Validate the fetched stock data
        
        Args:
            data (pd.DataFrame): Stock data to validate
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if required columns exist
        if not all(col in data.columns for col in required_columns):
            self.logger.error("Missing required columns in data")
            return False
        
        # Check for null values in critical columns
        if data[required_columns].isnull().any().any():
            self.logger.warning("Found null values in data")
        
        # Check if data has sufficient rows
        if len(data) < 50:
            self.logger.warning("Insufficient data points (less than 50)")
            return False
        
        return True
    
    def save_data(self, data, filename):
        """
        Save data to CSV file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Filename to save to
        """
        try:
            data.to_csv(filename)
            self.logger.info(f"Data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving data to {filename}: {str(e)}")
            raise
    
    def load_data(self, filename):
        """
        Load data from CSV file
        
        Args:
            filename (str): Filename to load from
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            self.logger.info(f"Data loaded from {filename}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data from {filename}: {str(e)}")
            raise
