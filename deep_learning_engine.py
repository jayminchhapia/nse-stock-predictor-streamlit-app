import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DeepLearningEngine:
    """Advanced Deep Learning Engine for Stock Price Prediction - FIXED VERSION"""
    
    def __init__(self, models_dir='dl_models'):
        self.models_dir = models_dir
        self.scalers_dir = 'dl_scalers'
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        # Model configurations
        self.sequence_length = 60  # Look back 60 days
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.target_column = 'Close'
    
    def prepare_data(self, symbol: str, period: str = '2y') -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare data for LSTM training with robust NaN handling"""
        try:
            # Fetch stock data
            ticker = yf.Ticker(f"{symbol}.NS" if not symbol.endswith('.NS') else symbol)
            data = ticker.history(period=period)
            
            if data.empty or len(data) < self.sequence_length + 50:
                raise ValueError(f"Insufficient data for {symbol}")
            
            print(f"📊 Initial data shape: {data.shape}")
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            print(f"📈 Data with indicators shape: {data.shape}")
            
            # FIXED: Robust NaN handling
            print("🧹 Cleaning NaN values...")
            
            # Forward fill first (carry last known value forward)
            data = data.fillna(method='ffill')
            
            # Backward fill for any remaining NaNs at the beginning
            data = data.fillna(method='bfill')
            
            # Drop any rows that still contain NaN
            data = data.dropna()
            
            print(f"✅ Cleaned data shape: {data.shape}")
            
            # Check if we have enough data after cleaning
            if len(data) < self.sequence_length + 30:
                raise ValueError(f"Insufficient data after cleaning for {symbol}. Need at least {self.sequence_length + 30}, got {len(data)}")
            
            # Select features for training
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                             'SMA_20', 'EMA_12', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
            
            # Only use columns that exist
            available_features = [col for col in feature_columns if col in data.columns]
            
            print(f"🎯 Selected features: {available_features}")
            
            # Extract feature data
            feature_data = data[available_features].values
            
            # Final NaN check and imputation if needed
            if np.isnan(feature_data).any():
                print("⚠️ Still contains NaN after cleaning - using imputer")
                imputer = SimpleImputer(strategy='mean')
                feature_data = imputer.fit_transform(feature_data)
            
            print(f"📊 Final feature data shape: {feature_data.shape}")
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(feature_data)
            
            # Verify no NaN in scaled data
            if np.isnan(scaled_data).any():
                raise ValueError("NaN values found in scaled data!")
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, data['Close'].values)
            
            print(f"✅ Created sequences - X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y, scaler
            
        except Exception as e:
            print(f"❌ Error preparing data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with proper NaN handling"""
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            
            # Exponential Moving Average
            df['EMA_12'] = df['Close'].ewm(span=12, min_periods=1).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, min_periods=1).mean()
            
            # RSI with minimum periods
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)  # Add small epsilon to prevent division by zero
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            bb_window = 20
            df['BB_Middle'] = df['Close'].rolling(window=bb_window, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=bb_window, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-8)
            
            # Price momentum
            df['Price_Change'] = df['Close'].pct_change()
            df['Volatility'] = df['Price_Change'].rolling(window=20, min_periods=1).std()
            
            # Fill any remaining NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Final safety check - fill with zeros if any NaN remains
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return data
    
    def _create_sequences(self, scaled_data: np.ndarray, target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        # Find the close price column index (should be index 3)
        close_index = 3
        
        for i in range(self.sequence_length, len(scaled_data)):
            # X: sequence of past data points
            X.append(scaled_data[i-self.sequence_length:i])
            # y: next day's closing price (normalized)
            y.append(scaled_data[i, close_index])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            # First LSTM layer with dropout
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            
            # Second LSTM layer
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            
            # Third LSTM layer
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def train_lstm_model(self, symbol: str, epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM model for a given symbol"""
        try:
            print(f"🔄 Training LSTM model for {symbol}...")
            
            # Prepare data
            X, y, scaler = self.prepare_data(symbol)
            if X is None or len(X) < 100:
                return {'success': False, 'error': 'Insufficient data for training'}
            
            # Split data
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            print(f"📊 Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            
            # Build model
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate model
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_mae = mean_absolute_error(y_test, test_predictions)
            
            # Save model and scaler
            model_path = os.path.join(self.models_dir, f"{symbol}_lstm_model.h5")
            scaler_path = os.path.join(self.scalers_dir, f"{symbol}_lstm_scaler.pkl")
            
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            print(f"✅ LSTM model trained for {symbol}")
            print(f"📈 Test RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}")
            
            return {
                'success': True,
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'model_path': model_path,
                'scaler_path': scaler_path,
                'training_history': history.history
            }
            
        except Exception as e:
            print(f"❌ Error training LSTM model for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def load_lstm_model(self, symbol: str) -> Tuple[Optional[Sequential], Optional[MinMaxScaler]]:
        """Load trained LSTM model and scaler"""
        try:
            model_path = os.path.join(self.models_dir, f"{symbol}_lstm_model.h5")
            scaler_path = os.path.join(self.scalers_dir, f"{symbol}_lstm_scaler.pkl")
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None, None
            
            model = keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            return model, scaler
            
        except Exception as e:
            print(f"Error loading LSTM model for {symbol}: {e}")
            return None, None
    
    def predict_lstm(self, symbol: str, days_ahead: int = 1) -> Dict:
        """Make predictions using trained LSTM model"""
        try:
            model, scaler = self.load_lstm_model(symbol)
            if model is None or scaler is None:
                return {'success': False, 'error': 'Model not found. Please train first.'}
            
            # Get recent data
            ticker = yf.Ticker(f"{symbol}.NS" if not symbol.endswith('.NS') else symbol)
            data = ticker.history(period='3mo')
            
            if len(data) < self.sequence_length:
                return {'success': False, 'error': 'Insufficient recent data'}
            
            # Prepare data
            data = self._add_technical_indicators(data)
            
            # Select the same features used during training
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                             'SMA_20', 'EMA_12', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
            available_features = [col for col in feature_columns if col in data.columns]
            feature_data = data[available_features].values
            
            # Scale data
            scaled_data = scaler.transform(feature_data)
            
            # Get last sequence
            last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, len(available_features))
            
            # Make prediction
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            # Get current price for comparison
            current_price = data['Close'].iloc[-1]
            
            # Denormalize the prediction
            dummy_array = np.zeros((1, len(available_features)))
            dummy_array[0, 3] = prediction  # Close price is at index 3
            predicted_price = scaler.inverse_transform(dummy_array)[0, 3]
            
            # Calculate metrics
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Determine recommendation
            if price_change_pct > 2:
                recommendation = "STRONG BUY"
            elif price_change_pct > 0.5:
                recommendation = "BUY"
            elif price_change_pct < -2:
                recommendation = "STRONG SELL"
            elif price_change_pct < -0.5:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'success': True,
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'recommendation': recommendation,
                'model_type': 'LSTM',
                'prediction_horizon': f'{days_ahead} day(s)',
                'confidence': min(abs(price_change_pct) * 10, 100)
            }
            
        except Exception as e:
            print(f"Error predicting with LSTM for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def lstm_model_exists(self, symbol: str) -> bool:
        """Check if LSTM model exists for symbol"""
        model_path = os.path.join(self.models_dir, f"{symbol}_lstm_model.h5")
        scaler_path = os.path.join(self.scalers_dir, f"{symbol}_lstm_scaler.pkl")
        return os.path.exists(model_path) and os.path.exists(scaler_path)
    
    def get_model_info(self, symbol: str) -> Dict:
        """Get information about trained models"""
        model_info = {
            'symbol': symbol,
            'lstm_available': self.lstm_model_exists(symbol),
            'model_files': []
        }
        
        if model_info['lstm_available']:
            model_path = os.path.join(self.models_dir, f"{symbol}_lstm_model.h5")
            if os.path.exists(model_path):
                stat = os.stat(model_path)
                model_info['model_files'].append({
                    'type': 'LSTM',
                    'path': model_path,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M')
                })
        
        return model_info
    
    def get_all_trained_models(self) -> List[str]:
        """Get list of all symbols with trained LSTM models"""
        trained_symbols = []
        
        if not os.path.exists(self.models_dir):
            return trained_symbols
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('_lstm_model.h5'):
                symbol = filename.replace('_lstm_model.h5', '')
                if self.lstm_model_exists(symbol):
                    trained_symbols.append(symbol)
        
        return trained_symbols

# Global instance
dl_engine = DeepLearningEngine()
