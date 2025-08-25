import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from data_engine import StockDataEngine

class MLModelTrainer:
    """ML Model Training and Management System"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.data_engine = StockDataEngine()
        os.makedirs(models_dir, exist_ok=True)
    
    def train_stock_model(self, symbol, analysis_type='next_day', period='1y'):
        """Train ML model for a specific stock"""
        try:
            print(f"🚀 Training {analysis_type} model for {symbol}...")
            
            # Determine prediction horizon
            horizon = 1 if analysis_type == 'next_day' else 5
            
            # Prepare training data
            features, labels, raw_data = self.data_engine.prepare_training_data(
                symbol=symbol, 
                period=period, 
                prediction_horizon=horizon
            )
            
            if features is None or len(features) < 50:
                print(f"❌ Insufficient data for {symbol}")
                return False, 0.0
            
            # Prepare data for training
            X = features.values
            y = labels.values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            accuracies = []
            best_model = None
            best_accuracy = 0
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                accuracies.append(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
            
            # Save best model and scaler
            model_path = os.path.join(self.models_dir, f"{symbol}_{analysis_type}")
            joblib.dump(best_model, f"{model_path}_model.pkl")
            joblib.dump(scaler, f"{model_path}_scaler.pkl")
            
            avg_accuracy = np.mean(accuracies)
            print(f"✅ {symbol} model trained: {avg_accuracy:.3f} accuracy")
            
            return True, avg_accuracy
            
        except Exception as e:
            print(f"❌ Training failed for {symbol}: {e}")
            return False, 0.0
    
    def train_multiple_stocks(self, symbols, analysis_type='next_day'):
        """Train models for multiple stocks"""
        results = {}
        
        print(f"🔄 Training {analysis_type} models for {len(symbols)} stocks...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n📊 Progress: {i}/{len(symbols)} - {symbol}")
            
            success, accuracy = self.train_stock_model(symbol, analysis_type)
            results[symbol] = {
                'success': success,
                'accuracy': accuracy if success else 0.0
            }
        
        # Summary
        successful = sum(1 for r in results.values() if r['success'])
        avg_accuracy = np.mean([r['accuracy'] for r in results.values() if r['success']])
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"✅ Successful: {successful}/{len(symbols)} stocks")
        print(f"📈 Average Accuracy: {avg_accuracy:.3f}")
        
        return results
    
    def load_model(self, symbol, analysis_type='next_day'):
        """Load trained model and scaler"""
        try:
            model_path = os.path.join(self.models_dir, f"{symbol}_{analysis_type}")
            model = joblib.load(f"{model_path}_model.pkl")
            scaler = joblib.load(f"{model_path}_scaler.pkl")
            return model, scaler
        except Exception as e:
            return None, None
    
    def predict_stock(self, symbol, analysis_type='next_day'):
        """Make prediction for a stock using trained model"""
        try:
            # Load model
            model, scaler = self.load_model(symbol, analysis_type)
            if model is None:
                return None
            
            # Get latest data and features
            horizon = 1 if analysis_type == 'next_day' else 5
            features, _, raw_data = self.data_engine.prepare_training_data(
                symbol=symbol, 
                period='6mo', 
                prediction_horizon=horizon
            )
            
            if features is None or len(features) == 0:
                return None
            
            # Use latest data point for prediction
            latest_features = features.iloc[-1:].values
            features_scaled = scaler.transform(latest_features)
            
            # Predict
            probability = model.predict_proba(features_scaled)[0][1]  # Probability of price going up
            prediction = model.predict(features_scaled)[0]
            
            # Get current price info
            current_price = raw_data['Close'].iloc[-1]
            prev_close = raw_data['Close'].iloc[-2] if len(raw_data) >= 2 else current_price
            
            return {
                'symbol': symbol,
                'prediction': prediction,
                'probability': probability,
                'confidence': max(probability, 1-probability),
                'current_price': current_price,
                'change_pct': ((current_price - prev_close) / prev_close * 100),
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
            return None

def train_nifty50_models():
    """Train models for top Nifty 50 stocks"""
    trainer = MLModelTrainer()
    
    # Top liquid Nifty 50 stocks
    stocks = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'BHARTIARTL', 'WIPRO', 'MARUTI', 'LT', 'AXISBANK'
    ]
    
    # Train next day models
    print("🚀 Training Next Day Models...")
    next_day_results = trainer.train_multiple_stocks(stocks, 'next_day')
    
    # Train short term models
    print("\n🚀 Training Short Term Models...")
    short_term_results = trainer.train_multiple_stocks(stocks, 'short_term')
    
    return next_day_results, short_term_results

if __name__ == "__main__":
    train_nifty50_models()
