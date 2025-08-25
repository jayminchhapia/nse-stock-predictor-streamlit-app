import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StockDataEngine:
    """Advanced Stock Data Collection and Feature Engineering System"""
    
    def __init__(self):
        self.data_cache = {}
        self.feature_cache = {}
    
    def fetch_stock_data(self, symbol, period='1y', interval='1d'):
        """Fetch comprehensive historical stock data from Yahoo Finance"""
        try:
            ticker = symbol if symbol.endswith('.NS') else symbol + '.NS'
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            
            if df.empty:
                print(f"No data available for {symbol}")
                return None
            
            df['symbol'] = symbol
            df['fetch_time'] = datetime.now()
            df['daily_return'] = df['Close'].pct_change()
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_advanced_features(self, df):
        """Calculate 18+ advanced technical indicators"""
        try:
            features_df = df.copy()
            
            # Basic Returns and Volatility
            features_df['returns'] = features_df['Close'].pct_change()
            features_df['log_returns'] = np.log(features_df['Close'] / features_df['Close'].shift(1))
            features_df['volatility_5'] = features_df['returns'].rolling(5).std()
            features_df['volatility_20'] = features_df['returns'].rolling(20).std()
            
            # Moving Averages
            features_df['ma5'] = features_df['Close'].rolling(5).mean()
            features_df['ma20'] = features_df['Close'].rolling(20).mean()
            features_df['ma50'] = features_df['Close'].rolling(50).mean()
            
            # Price to MA Ratios
            features_df['price_ma5_ratio'] = features_df['Close'] / features_df['ma5']
            features_df['price_ma20_ratio'] = features_df['Close'] / features_df['ma20']
            features_df['price_ma50_ratio'] = features_df['Close'] / features_df['ma50']
            
            # RSI (Relative Strength Index)
            delta = features_df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema12 = features_df['Close'].ewm(span=12).mean()
            ema26 = features_df['Close'].ewm(span=26).mean()
            features_df['macd'] = ema12 - ema26
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Bollinger Bands
            rolling_std = features_df['Close'].rolling(20).std()
            features_df['bollinger_upper'] = features_df['ma20'] + (rolling_std * 2)
            features_df['bollinger_lower'] = features_df['ma20'] - (rolling_std * 2)
            features_df['bollinger_position'] = (features_df['Close'] - features_df['bollinger_lower']) / (features_df['bollinger_upper'] - features_df['bollinger_lower'])
            
            # Volume Analysis
            features_df['volume_ma20'] = features_df['Volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_ma20']
            
            # Momentum Indicators
            features_df['momentum_5'] = features_df['Close'] / features_df['Close'].shift(5) - 1
            features_df['momentum_10'] = features_df['Close'] / features_df['Close'].shift(10) - 1
            features_df['momentum_20'] = features_df['Close'] / features_df['Close'].shift(20) - 1
            
            # Support and Resistance
            features_df['support_20'] = features_df['Low'].rolling(20).min()
            features_df['resistance_20'] = features_df['High'].rolling(20).max()
            features_df['support_distance'] = (features_df['Close'] - features_df['support_20']) / features_df['support_20']
            features_df['resistance_distance'] = (features_df['resistance_20'] - features_df['Close']) / features_df['Close']
            
            # Select final features for ML
            feature_columns = [
                'returns', 'log_returns', 'volatility_5', 'volatility_20',
                'price_ma5_ratio', 'price_ma20_ratio', 'price_ma50_ratio',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bollinger_position', 'volume_ratio',
                'momentum_5', 'momentum_10', 'momentum_20',
                'support_distance', 'resistance_distance'
            ]
            
            ml_features = features_df[feature_columns].copy()
            ml_features = ml_features.dropna()
            
            return ml_features, features_df
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None, None
    
    def generate_labels(self, df, prediction_horizon=1):
        """Generate prediction labels for ML training"""
        try:
            future_price = df['Close'].shift(-prediction_horizon)
            current_price = df['Close']
            labels = (future_price > current_price).astype(int)
            return labels.dropna()
            
        except Exception as e:
            print(f"Error generating labels: {e}")
            return None
    
    def prepare_training_data(self, symbol, period='1y', prediction_horizon=1):
        """Complete pipeline to prepare training data"""
        try:
            # Fetch data
            raw_data = self.fetch_stock_data(symbol, period=period)
            if raw_data is None:
                return None, None, None
            
            # Calculate features
            features, processed_data = self.calculate_advanced_features(raw_data)
            if features is None:
                return None, None, None
            
            # Generate labels
            labels = self.generate_labels(raw_data, prediction_horizon)
            if labels is None:
                return None, None, None
            
            # Align features and labels
            common_index = features.index.intersection(labels.index)
            features_aligned = features.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            return features_aligned, labels_aligned, raw_data
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None, None

if __name__ == "__main__":
    # Test the data engine
    engine = StockDataEngine()
    features, labels, raw_data = engine.prepare_training_data('RELIANCE', '6mo', 1)
    if features is not None:
        print(f"Features shape: {features.shape}")
        print(f"Labels length: {len(labels)}")
        print("Sample features:")
        print(features.head())
    else:
        print("Failed to prepare data")
