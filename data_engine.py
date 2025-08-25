import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

class StockDataEngine:
    """Fixed Stock Data Engine - Handles MultiIndex Issues"""
    
    def __init__(self):
        self.data_cache = {}
    
    def fetch_stock_data(self, symbol, period='6mo', interval='1d', max_retries=2):
        """Fixed data fetching with proper column handling"""
        
        ticker_variants = [f"{symbol}.NS", f"{symbol}.BO", symbol]
        
        for attempt in range(max_retries):
            for ticker in ticker_variants:
                try:
                    print(f"🔍 Attempt {attempt + 1}: Fetching {ticker} (period: {period})")
                    
                    df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=20)
                    
                    if not df.empty and len(df) >= 10:
                        print(f"✅ Successfully fetched {len(df)} rows for {ticker}")
                        
                        # FIX: Handle MultiIndex columns from yfinance
                        if isinstance(df.columns, pd.MultiIndex):
                            # Flatten MultiIndex columns by taking the first level
                            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                        
                        # Ensure we have the required columns
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            print(f"⚠️ Missing columns {missing_cols}, trying next ticker...")
                            continue
                        
                        # Add symbol column
                        df['symbol'] = symbol
                        
                        print(f"📊 Columns: {list(df.columns)}")
                        return df
                    else:
                        print(f"⚠️ Empty data for {ticker}")
                        
                except Exception as e:
                    print(f"❌ Error with {ticker}: {str(e)[:50]}")
            
            if attempt < max_retries - 1:
                time.sleep(1)
        
        # Try fallback periods
        for fallback in ['3mo', '1mo']:
            try:
                print(f"🔄 Fallback: {symbol}.NS with {fallback}")
                df = yf.download(f"{symbol}.NS", period=fallback, progress=False, timeout=20)
                if not df.empty and len(df) >= 5:
                    # FIX: Handle MultiIndex columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    
                    print(f"✅ Fallback success: {len(df)} rows")
                    df['symbol'] = symbol
                    return df
            except:
                continue
        
        print(f"❌ Failed to fetch data for {symbol}")
        return None
    
    def calculate_advanced_features(self, df):
        """Fixed feature calculation ensuring Series operations"""
        try:
            if df is None or df.empty:
                print("Cannot calculate features: DataFrame is empty")
                return None, None
                
            print(f"🔧 Computing features for {len(df)} data points...")
            
            # Work on copy and ensure proper column access
            features_df = df.copy()
            
            # Verify essential columns exist
            required_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
            missing_columns = [col for col in required_columns if col not in features_df.columns]
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return None, None
            
            print("✅ All required columns present")
            
            # FIX: Ensure all operations work on Series (single columns)
            close_series = features_df['Close']
            volume_series = features_df['Volume']
            high_series = features_df['High']
            low_series = features_df['Low']
            open_series = features_df['Open']
            
            # Verify these are Series, not DataFrames
            if not isinstance(close_series, pd.Series):
                print(f"❌ Close is not Series: {type(close_series)}")
                return None, None
            
            print("✅ Column types verified as Series")
            
            # 1. Basic returns - Direct Series operations
            features_df['returns'] = close_series.pct_change()
            features_df['log_returns'] = np.log(close_series / close_series.shift(1))
            
            # 2. Volatility
            returns_series = features_df['returns']
            features_df['volatility_5'] = returns_series.rolling(5).std()
            features_df['volatility_20'] = returns_series.rolling(20).std()
            
            # 3. Moving averages
            print("  📊 Calculating moving averages...")
            features_df['ma5'] = close_series.rolling(5).mean()
            features_df['ma20'] = close_series.rolling(20).mean()
            features_df['ma50'] = close_series.rolling(50).mean()
            
            # 4. Price ratios - Direct Series division
            print("  📊 Calculating price ratios...")
            ma5_series = features_df['ma5']
            ma20_series = features_df['ma20']
            ma50_series = features_df['ma50']
            
            features_df['price_ma5_ratio'] = close_series / ma5_series
            features_df['price_ma20_ratio'] = close_series / ma20_series
            features_df['price_ma50_ratio'] = close_series / ma50_series
            
            # 5. RSI
            print("  📊 Calculating RSI...")
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # 6. MACD
            print("  📊 Calculating MACD...")
            ema12 = close_series.ewm(span=12).mean()
            ema26 = close_series.ewm(span=26).mean()
            macd = ema12 - ema26
            features_df['macd'] = macd
            features_df['macd_signal'] = macd.ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # 7. Bollinger Bands
            print("  📊 Calculating Bollinger Bands...")
            rolling_std = close_series.rolling(20).std()
            bollinger_upper = ma20_series + (rolling_std * 2)
            bollinger_lower = ma20_series - (rolling_std * 2)
            features_df['bollinger_position'] = (close_series - bollinger_lower) / (bollinger_upper - bollinger_lower)
            
            # 8. Volume
            print("  📊 Calculating volume features...")
            volume_ma20 = volume_series.rolling(20).mean()
            features_df['volume_ma20'] = volume_ma20
            features_df['volume_ratio'] = volume_series / volume_ma20
            
            # 9. Momentum
            print("  📊 Calculating momentum...")
            features_df['momentum_5'] = close_series / close_series.shift(5) - 1
            features_df['momentum_10'] = close_series / close_series.shift(10) - 1
            features_df['momentum_20'] = close_series / close_series.shift(20) - 1
            
            # 10. Support/Resistance
            print("  📊 Calculating support/resistance...")
            support_20 = low_series.rolling(20).min()
            resistance_20 = high_series.rolling(20).max()
            features_df['support_20'] = support_20
            features_df['resistance_20'] = resistance_20
            features_df['support_distance'] = (close_series - support_20) / support_20
            features_df['resistance_distance'] = (resistance_20 - close_series) / close_series
            
            # Select final ML features
            ml_feature_columns = [
                'returns', 'log_returns', 'volatility_5', 'volatility_20',
                'price_ma5_ratio', 'price_ma20_ratio', 'price_ma50_ratio',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bollinger_position', 'volume_ratio',
                'momentum_5', 'momentum_10', 'momentum_20',
                'support_distance', 'resistance_distance'
            ]
            
            # Create clean feature DataFrame
            available_features = [col for col in ml_feature_columns if col in features_df.columns]
            print(f"  📊 Available features: {len(available_features)}/{len(ml_feature_columns)}")
            
            if len(available_features) == 0:
                print("❌ No features available")
                return None, None
            
            ml_features = features_df[available_features].copy()
            
            # Remove invalid values
            ml_features = ml_features.replace([np.inf, -np.inf], np.nan)
            ml_features = ml_features.dropna()
            
            if len(ml_features) == 0:
                print("❌ No valid features after cleaning")
                return None, None
            
            print(f"✅ Generated {len(available_features)} features for {len(ml_features)} data points")
            return ml_features, features_df
            
        except Exception as e:
            print(f"❌ Feature calculation error: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print("❌ Traceback:", traceback.format_exc()[-300:])
            return None, None
    
    def generate_labels(self, df, prediction_horizon=1):
        """Generate binary labels for price direction"""
        try:
            close_series = df['Close']
            future_price = close_series.shift(-prediction_horizon)
            current_price = close_series
            labels = (future_price > current_price).astype(int)
            labels_clean = labels.dropna()
            
            if len(labels_clean) > 0:
                print(f"🎯 Generated {len(labels_clean)} labels (UP: {labels_clean.sum()}, DOWN: {len(labels_clean)-labels_clean.sum()})")
                return labels_clean
            else:
                print("❌ No valid labels generated")
                return None
                
        except Exception as e:
            print(f"❌ Label generation error: {str(e)}")
            return None
    
    def prepare_training_data(self, symbol, period='6mo', prediction_horizon=1):
        """Main pipeline for training data preparation"""
        try:
            print(f"🔄 Preparing training data for {symbol}...")
            
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
            
            # Align data
            common_index = features.index.intersection(labels.index)
            if len(common_index) == 0:
                print("❌ No common timestamps")
                return None, None, None
            
            features_aligned = features.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            print(f"✅ Training data ready: {len(features_aligned)} samples with {len(features.columns)} features")
            return features_aligned, labels_aligned, raw_data
            
        except Exception as e:
            print(f"❌ Preparation error: {str(e)}")
            return None, None, None

if __name__ == "__main__":
    engine = StockDataEngine()
    features, labels, data = engine.prepare_training_data('RELIANCE', '6mo', 1)
    if features is not None:
        print(f"Success! Features: {features.shape}, Labels: {len(labels)}")
        print("\nFirst few features:")
        print(features.head())
    else:
        print("Failed to prepare data")
