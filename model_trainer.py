import os
import joblib
import numpy as np
from deep_learning_engine import dl_engine
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from data_engine import StockDataEngine

class MLModelTrainer:
    """ML Model Training with Auto-Training Capabilities"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.data_engine = StockDataEngine()
        os.makedirs(models_dir, exist_ok=True)
    
    def model_exists(self, symbol, analysis_type='next_day'):
        """Check if trained model exists for given symbol and analysis type"""
        model_path = os.path.join(self.models_dir, f"{symbol}_{analysis_type}")
        model_file = f"{model_path}_model.pkl"
        scaler_file = f"{model_path}_scaler.pkl"
        return os.path.exists(model_file) and os.path.exists(scaler_file)
    
    def train_stock_model(self, symbol, analysis_type='next_day', period='1y', show_progress=True):
        """Enhanced training with better error handling and progress tracking"""
        try:
            if show_progress:
                print(f"🚀 Training {analysis_type.replace('_', ' ')} model for {symbol}...")
            
            # Determine prediction horizon
            horizon = 1 if analysis_type == 'next_day' else 5
            
            # Prepare training data
            features, labels, raw_data = self.data_engine.prepare_training_data(
                symbol=symbol, 
                period=period, 
                prediction_horizon=horizon
            )
            
            if features is None or len(features) < 50:
                error_msg = f"❌ Insufficient data for {symbol} (got {len(features) if features is not None else 0} samples, need 50+)"
                if show_progress:
                    print(error_msg)
                return False, 0.0
            
            if show_progress:
                print(f"✅ Data prepared: {len(features)} samples with {len(features.columns)} features")
            
            # Prepare data for training
            X = features.values
            y = labels.values
            
            # Scale features and train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for faster training
            accuracies = []
            best_model = None
            best_accuracy = 0
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model with reduced complexity for faster training
                model = RandomForestClassifier(
                    n_estimators=50,  # Reduced from 100 for speed
                    max_depth=8,      # Reduced from 10 for speed
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=1  # Single thread to avoid issues on cloud
                )
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                accuracies.append(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                
                if show_progress:
                    print(f"  Fold {fold + 1}: {accuracy:.3f} accuracy")
            
            # Save best model and scaler
            model_path = os.path.join(self.models_dir, f"{symbol}_{analysis_type}")
            joblib.dump(best_model, f"{model_path}_model.pkl")
            joblib.dump(scaler, f"{model_path}_scaler.pkl")
            
            avg_accuracy = np.mean(accuracies)
            success_msg = f"✅ {symbol} model trained successfully: {avg_accuracy:.3f} average accuracy"
            
            if show_progress:
                print(success_msg)
            
            return True, avg_accuracy
            
        except Exception as e:
            error_msg = f"❌ Training failed for {symbol}: {str(e)}"
            if show_progress:
                print(error_msg)
            return False, 0.0
    
    def load_model(self, symbol, analysis_type='next_day'):
        """Load trained model and scaler with error handling"""
        try:
            model_path = os.path.join(self.models_dir, f"{symbol}_{analysis_type}")
            model_file = f"{model_path}_model.pkl"
            scaler_file = f"{model_path}_scaler.pkl"
            
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                return None, None
            
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            return model, scaler
            
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
            return None, None
    
    def get_trained_models_status(self):
        """Get status of all trained models - THIS WAS THE MISSING METHOD"""
        if not os.path.exists(self.models_dir):
            return {}
        
        models = {}
        try:
            for file in os.listdir(self.models_dir):
                if file.endswith('_model.pkl'):
                    # Parse filename: SYMBOL_ANALYSISTYPE_model.pkl
                    parts = file.replace('_model.pkl', '').split('_')
                    if len(parts) >= 2:
                        symbol = '_'.join(parts[:-1])  # Handle symbols with underscores
                        analysis_type = parts[-1]
                        
                        if symbol not in models:
                            models[symbol] = []
                        models[symbol].append(analysis_type)
        except Exception as e:
            print(f"Error reading models directory: {e}")
            return {}
        
        return models
    
    def predict_stock_with_auto_training(self, symbol, analysis_type='next_day'):
        """Enhanced prediction with automatic training if model doesn't exist"""
        try:
            # Check if model exists, if not, train it
            if not self.model_exists(symbol, analysis_type):
                print(f"🔄 Model for {symbol} ({analysis_type.replace('_', ' ')}) not found. Training automatically...")
                
                success, accuracy = self.train_stock_model(symbol, analysis_type, show_progress=True)
                
                if not success:
                    print(f"Failed to train model for {symbol}. Please check data availability.")
                    return None
                
                print(f"Model trained with {accuracy:.1%} accuracy")
            
            # Load model
            model, scaler = self.load_model(symbol, analysis_type)
            if model is None or scaler is None:
                print(f"Failed to load model for {symbol} even after training attempt.")
                return None
            
            # Get latest data and features
            horizon = 1 if analysis_type == 'next_day' else 5
            features, _, raw_data = self.data_engine.prepare_training_data(
                symbol=symbol, 
                period='6mo', 
                prediction_horizon=horizon
            )
            
            if features is None or len(features) == 0:
                print(f"Unable to fetch current data for {symbol}")
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
            
            result = {
                'symbol': symbol,
                'prediction': prediction,
                'probability': probability,
                'confidence': max(probability, 1-probability),
                'current_price': current_price,
                'change_pct': ((current_price - prev_close) / prev_close * 100),
                'analysis_type': analysis_type,
                'model_trained': True
            }
            
            return result
            
        except Exception as e:
            print(f"Prediction error for {symbol}: {str(e)}")
            return None

    def train_ensemble_model(self, symbol: str, analysis_type: str = 'next_day', use_lstm: bool = True):
        """Train ensemble model combining Random Forest + LSTM"""
        try:
            print(f"🔄 Training ensemble model for {symbol}...")
            
            results = {'rf_success': False, 'lstm_success': False}
            
            # Train Random Forest (existing method)
            rf_success, rf_accuracy = self.train_stock_model(symbol, analysis_type, show_progress=False)
            results['rf_success'] = rf_success
            results['rf_accuracy'] = rf_accuracy
            
            # Train LSTM if requested
            if use_lstm:
                lstm_result = dl_engine.train_lstm_model(symbol)
                results['lstm_success'] = lstm_result['success']
                if lstm_result['success']:
                    results['lstm_rmse'] = lstm_result['test_rmse']
                    results['lstm_mae'] = lstm_result['test_mae']
            
            return results
            
        except Exception as e:
            print(f"Error training ensemble model: {e}")
            return {'rf_success': False, 'lstm_success': False, 'error': str(e)}
    
    def predict_ensemble(self, symbol: str, analysis_type: str = 'next_day') -> Dict:
        """Make predictions using both RF and LSTM models"""
        try:
            predictions = {}
            
            # Random Forest prediction
            if self.model_exists(symbol, analysis_type):
                model, scaler = self.load_model(symbol, analysis_type)
                features, _, _ = self.data_engine.prepare_training_data(symbol, '6mo', 1)
                
                if features is not None and model is not None:
                    latest_features = features.iloc[-1:].values
                    features_scaled = scaler.transform(latest_features)
                    rf_probability = model.predict_proba(features_scaled)[0][1]
                    
                    predictions['rf'] = {
                        'probability': rf_probability,
                        'recommendation': self._get_recommendation(rf_probability),
                        'confidence': max(rf_probability, 1-rf_probability)
                    }
            
            # LSTM prediction
            if dl_engine.lstm_model_exists(symbol):
                lstm_result = dl_engine.predict_lstm(symbol)
                if lstm_result['success']:
                    predictions['lstm'] = {
                        'predicted_price': lstm_result['predicted_price'],
                        'current_price': lstm_result['current_price'],
                        'price_change_pct': lstm_result['price_change_pct'],
                        'recommendation': lstm_result['recommendation'],
                        'confidence': lstm_result['confidence']
                    }
            
            # Ensemble recommendation (if both models available)
            if 'rf' in predictions and 'lstm' in predictions:
                ensemble_rec = self._combine_recommendations(
                    predictions['rf']['recommendation'],
                    predictions['lstm']['recommendation']
                )
                predictions['ensemble'] = {
                    'recommendation': ensemble_rec,
                    'confidence': (predictions['rf']['confidence'] + predictions['lstm']['confidence']) / 2
                }
            
            return predictions
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return {'error': str(e)}
    
    def _get_recommendation(self, probability: float) -> str:
        """Convert probability to recommendation"""
        if probability >= 0.75:
            return "STRONG BUY"
        elif probability >= 0.6:
            return "BUY"
        elif probability <= 0.25:
            return "STRONG SELL"
        elif probability <= 0.4:
            return "SELL"
        else:
            return "HOLD"
    
    def _combine_recommendations(self, rf_rec: str, lstm_rec: str) -> str:
        """Combine recommendations from different models"""
        buy_signals = ["STRONG BUY", "BUY"]
        sell_signals = ["STRONG SELL", "SELL"]
        
        if rf_rec in buy_signals and lstm_rec in buy_signals:
            return "STRONG BUY"
        elif rf_rec in sell_signals and lstm_rec in sell_signals:
            return "STRONG SELL"
        elif rf_rec in buy_signals or lstm_rec in buy_signals:
            return "BUY"
        elif rf_rec in sell_signals or lstm_rec in sell_signals:
            return "SELL"
        else:
            return "HOLD"
if __name__ == "__main__":
    # Test the model trainer
    trainer = MLModelTrainer()
    
    # Test training
    success, accuracy = trainer.train_stock_model('RELIANCE', 'next_day')
    print(f"Training success: {success}, Accuracy: {accuracy}")
    
    # Test model status
    status = trainer.get_trained_models_status()
    print(f"Trained models: {status}")
