import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import streamlit as st
import time
warnings.filterwarnings('ignore')

class StockDataEngine:
    """Enhanced Stock Data Engine with Robust Error Handling"""
    
    def __init__(self):
        self.data_cache = {}
        self.feature_cache = {}
    
    def fetch_stock_data_robust(self, symbol, period='1y', interval='1d', max_retries=3):
        """Robust data fetching with multiple fallback strategies"""
        
        # Try different ticker formats
        ticker_variants = [
            f"{symbol}.NS",  # NSE format
            f"{symbol}.BO",  # BSE format  
            symbol,          # Direct symbol
        ]
        
        for attempt in range(max_retries):
            for ticker in ticker_variants:
                try:
                    st.info(f"🔍 Attempt {attempt + 1}: Fetching {ticker} (period: {period})")
                    
                    # Try with different parameters
                    df = yf.download(
                        ticker, 
                        period=period, 
                        interval=interval, 
                        progress=False,
                        auto_adjust=False,
                        timeout=30
                    )
                    
                    if not df.empty and len(df) >= 10:  # Need at least 10 data points
                        st.success(f"✅ Successfully fetched {len(df)} rows for {ticker}")
                        
                        # Add metadata
                        df['symbol'] = symbol
                        df['fetch_time'] = datetime.now()
                        df['daily_return'] = df['Close'].pct_change()
                        
                        # Cache successful fetch
                        self.data_cache[f"{symbol}_{period}"] = df.copy()
                        
                        return df
                    else:
                        st.warning(f"⚠️ Empty data for {ticker}, trying next variant...")
                        
                except Exception as e:
                    st.warning(f"❌ Error with {ticker}: {str(e)[:100]}")
                    continue
            
            if attempt < max_retries - 1:
                st.info(f"⏳ Waiting 2 seconds before retry {attempt + 2}...")
                time.sleep(2)
        
        # If all attempts fail, try fallback periods
        st.warning(f"🔄 All standard attempts failed. Trying fallback periods...")
        
        fallback_periods = ['6mo', '3mo', '1mo', '5d']
        
        for fallback_period in fallback_periods:
            if fallback_period != period:  # Don't repeat same period
                try:
                    ticker = f"{symbol}.NS"  # Use most common format
                    st.info(f"🔄 Fallback: Trying {ticker} with {fallback_period} period")
                    
                    df = yf.download(
                        ticker, 
                        period=fallback_period, 
                        interval=interval, 
                        progress=False,
                        timeout=30
                    )
                    
                    if not df.empty and len(df) >= 5:  # Lower threshold for fallback
                        st.success(f"✅ Fallback successful: {len(df)} rows for {ticker}")
                        
                        df['symbol'] = symbol
                        df['fetch_time'] = datetime.now()
                        df['daily_return'] = df['Close'].pct_change()
                        
                        return df
                        
                except Exception as e:
                    st.warning(f"❌ Fallback {fallback_period} failed: {str(e)[:50]}")
                    continue
        
        # Final fallback with sample data (for demo purposes)
        st.error(f"❌ All data fetching attempts failed for {symbol}")
        st.info("💡 This might be due to:")
        st.info("   • Yahoo Finance API rate limits")
        st.info("   • Network restrictions on cloud deployment")  
        st.info("   • Market closure (weekends/holidays)")
        st.info("   • Invalid stock symbol")
        
        return None
    
    def fetch_stock_data(self, symbol, period='1y', interval='1d'):
        """Main fetch method with enhanced error handling"""
        return self.fetch_stock_data_robust(symbol, period, interval)
    
    def calculate_advanced_features(self, df):
        """Calculate advanced features with error handling"""
        try:
            if df is None or df.empty:
                st.error("Cannot calculate features: DataFrame is empty")
                return None, None
                
            st.info(f"🔧 Computing features for {len(df)} data points...")
            
            features_df = df.copy()
            
            # Ensure minimum data points
            if len(features_df) < 20:
                st.warning(f"⚠️ Limited data ({len(features_df)} points). Features may be less reliable.")
            
            # Basic Returns and Volatility
            features_df['returns'] = features_df['Close'].pct_change()
            features_df['log_returns'] = np.log(features_df['Close'] / features_df['Close'].shift(1))
            features_df['volatility_5'] = features_df['returns'].rolling(min(5, len(features_df)//4)).std()
            features_df['volatility_20'] = features_df['returns'].rolling(min(20, len(features_df)//2)).std()
            
            # Moving Averages (adapt to data size)
            ma5_window = min(5, len(features_df)//4)
            ma20_window = min(20, len(features_df)//2)
            ma50_window = min(50, len(features_df)*3//4)
            
            features_df['ma5'] = features_df['Close'].rolling(ma5_window).mean()
            features_df['ma20'] = features_df['Close'].rolling(ma20_window).mean()
            features_df['ma50'] = features_df['Close'].rolling(ma50_window).mean()
            
            # Price to MA Ratios
            features_df['price_ma5_ratio'] = features_df['Close'] / features_df['ma5']
            features_df['price_ma20_ratio'] = features_df['Close'] / features_df['ma20']
            features_df['price_ma50_ratio'] = features_df['Close'] / features_df['ma50']
            
            # RSI (adapt window size)
            rsi_window = min(14, len(features_df)//3)
            delta = features_df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(rsi_window).mean()
            avg_loss = loss.rolling(rsi_window).mean()
            rs = avg_gain / avg_loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12_span = min(12, len(features_df)//4)
            ema26_span = min(26, len(features_df)//2)
            ema12 = features_df['Close'].ewm(span=ema12_span).mean()
            ema26 = features_df['Close'].ewm(span=ema26_span).mean()
            features_df['macd'] = ema12 - ema26
            features_df['macd_signal'] = features_df['macd'].ewm(span=min(9, len(features_df)//5)).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Bollinger Bands
            bb_window = min(20, len(features_df)//2)
            rolling_std = features_df['Close'].rolling(bb_window).std()
            features_df['bollinger_upper'] = features_df['ma20'] + (rolling_std * 2)
            features_df['bollinger_lower'] = features_df['ma20'] - (rolling_std * 2)
            features_df['bollinger_position'] = (features_df['Close'] - features_df['bollinger_lower']) / (features_df['bollinger_upper'] - features_df['bollinger_lower'])
            
            # Volume Analysis
            vol_window = min(20, len(features_df)//2)
            features_df['volume_ma20'] = features_df['Volume'].rolling(vol_window).mean()
            features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_ma20']
            
            # Momentum Indicators
            mom5_window = min(5, len(features_df)//4)
            mom10_window = min(10, len(features_df)//3)
            mom20_window = min(20, len(features_df)//2)
            
            features_df['momentum_5'] = features_df['Close'] / features_df['Close'].shift(mom5_window) - 1
            features_df['momentum_10'] = features_df['Close'] / features_df['Close'].shift(mom10_window) - 1
            features_df['momentum_20'] = features_df['Close'] / features_df['Close'].shift(mom20_window) - 1
            
            # Support and Resistance
            sr_window = min(20, len(features_df)//2)
            features_df['support_20'] = features_df['Low'].rolling(sr_window).min()
            features_df['resistance_20'] = features_df['High'].rolling(sr_window).max()
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
            
            # Filter out columns that might not exist due to insufficient data
            available_columns = [col for col in feature_columns if col in features_df.columns]
            
            ml_features = features_df[available_columns].copy()
            ml_features = ml_features.dropna()
            
            if len(ml_features) == 0:
                st.error("❌ No valid feature rows after cleaning. Insufficient data.")
                return None, None
            
            st.success(f"✅ Generated {len(available_columns)} features for {len(ml_features)} data points")
            
            return ml_features, features_df
            
        except Exception as e:
            st.error(f"❌ Error calculating features: {str(e)}")
            return None, None
    
    def generate_labels(self, df, prediction_horizon=1):
        """Generate labels with error handling"""
        try:
            if df is None or df.empty:
                return None
                
            future_price = df['Close'].shift(-prediction_horizon)
            current_price = df['Close']
            labels = (future_price > current_price).astype(int)
            labels_clean = labels.dropna()
            
            if len(labels_clean) == 0:
                st.error("❌ No valid labels generated")
                return None
                
            st.info(f"🎯 Generated {len(labels_clean)} labels (UP: {labels_clean.sum()}, DOWN: {len(labels_clean)-labels_clean.sum()})")
            return labels_clean
            
        except Exception as e:
            st.error(f"❌ Error generating labels: {str(e)}")
            return None
    
    def prepare_training_data(self, symbol, period='1y', prediction_horizon=1):
        """Enhanced training data preparation with comprehensive error handling"""
        try:
            st.info(f"🔄 Preparing training data for {symbol}...")
            
            # Step 1: Fetch data
            raw_data = self.fetch_stock_data(symbol, period=period)
            if raw_data is None:
                st.error(f"❌ Could not fetch data for {symbol}")
                return None, None, None
            
            # Step 2: Calculate features
            features, processed_data = self.calculate_advanced_features(raw_data)
            if features is None:
                st.error(f"❌ Could not calculate features for {symbol}")
                return None, None, None
            
            # Step 3: Generate labels
            labels = self.generate_labels(raw_data, prediction_horizon)
            if labels is None:
                st.error(f"❌ Could not generate labels for {symbol}")
                return None, None, None
            
            # Step 4: Align features and labels
            common_index = features.index.intersection(labels.index)
            
            if len(common_index) == 0:
                st.error(f"❌ No common timestamps between features and labels for {symbol}")
                return None, None, None
            
            features_aligned = features.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            st.success(f"✅ Training data ready: {len(features_aligned)} samples with {len(features.columns)} features")
            
            return features_aligned, labels_aligned, raw_data
            
        except Exception as e:
            st.error(f"❌ Error preparing training data for {symbol}: {str(e)}")
            return None, None, None

if __name__ == "__main__":
    # Test the enhanced data engine
    engine = StockDataEngine()
    features, labels, raw_data = engine.prepare_training_data('RELIANCE', '6mo', 1)
    if features is not None:
        print(f"Features shape: {features.shape}")
        print(f"Labels length: {len(labels)}")
        print("Sample features:")
        print(features.head())
    else:
        print("Failed to prepare data")
