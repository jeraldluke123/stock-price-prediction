import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import talib
import logging

class FeatureEngineer:
    """
    Feature engineering for stock price prediction
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
    
    def add_technical_indicators(self, data):
        """
        Add technical indicators to the dataset
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        df = data.copy()
        
        try:
            # Price-based indicators
            df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
            df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            
            # Exponential Moving Averages
            df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
            df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(df['close'])
            
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            
            # Stochastic Oscillator
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            
            # Average True Range
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Volume indicators
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=10)
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])  # Accumulation/Distribution
            df['obv'] = talib.OBV(df['close'], df['volume'])  # On Balance Volume
            
            self.logger.info("Technical indicators added successfully")
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            # Fallback to simple indicators if talib fails
            df = self._add_simple_indicators(df)
        
        return df
    
    def _add_simple_indicators(self, data):
        """
        Add simple technical indicators without talib dependency
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with simple indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        
        # Volume indicators
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        
        # Volatility (rolling standard deviation)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Simple RSI approximation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        self.logger.info("Simple indicators added successfully")
        return df
    
    def add_lag_features(self, data, target_col='close', lags=[1, 2, 3, 5, 10]):
        """
        Add lagged features
        
        Args:
            data (pd.DataFrame): Stock data
            target_col (str): Column to create lags for
            lags (list): List of lag periods
        
        Returns:
            pd.DataFrame: Data with lag features
        """
        df = data.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        self.logger.info(f"Added lag features for periods: {lags}")
        return df
    
    def add_rolling_features(self, data, target_col='close', windows=[5, 10, 20]):
        """
        Add rolling statistical features
        
        Args:
            data (pd.DataFrame): Stock data
            target_col (str): Column to calculate rolling stats for
            windows (list): List of window sizes
        
        Returns:
            pd.DataFrame: Data with rolling features
        """
        df = data.copy()
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        self.logger.info(f"Added rolling features for windows: {windows}")
        return df
    
    def create_sequences(self, data, sequence_length=60, target_col='close'):
        """
        Create sequences for LSTM model
        
        Args:
            data (pd.DataFrame): Preprocessed stock data
            sequence_length (int): Length of input sequences
            target_col (str): Target column name
        
        Returns:
            tuple: (X, y) where X is sequences and y is targets
        """
        # Select features for training
        feature_cols = [col for col in data.columns if col not in ['symbol', target_col]]
        
        # Remove rows with NaN values
        clean_data = data.dropna()
        
        if len(clean_data) < sequence_length:
            raise ValueError(f"Not enough data points. Need at least {sequence_length}, got {len(clean_data)}")
        
        # Scale the features
        features = clean_data[feature_cols].values
        targets = clean_data[target_col].values
        
        # Fit scaler on features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        return X, y
    
    def prepare_data_for_lstm(self, data, sequence_length=60, test_size=0.2, target_col='close'):
        """
        Complete data preparation pipeline for LSTM
        
        Args:
            data (pd.DataFrame): Raw stock data
            sequence_length (int): Length of input sequences
            test_size (float): Proportion of data for testing
            target_col (str): Target column name
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        # Add all features
        df = self.add_technical_indicators(data)
        df = self.add_lag_features(df, target_col)
        df = self.add_rolling_features(df, target_col)
        
        # Create sequences
        X, y = self.create_sequences(df, sequence_length, target_col)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        self.logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test, self.scaler
    
    def inverse_transform_predictions(self, predictions, original_data, target_col='close'):
        """
        Inverse transform scaled predictions back to original scale
        
        Args:
            predictions (np.array): Scaled predictions
            original_data (pd.DataFrame): Original data for reference
            target_col (str): Target column name
        
        Returns:
            np.array: Predictions in original scale
        """
        # This is a simplified approach - in practice, you'd need to handle this more carefully
        # depending on how you scaled your target variable
        return predictions
