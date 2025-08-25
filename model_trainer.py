import os
import joblib
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from data_engine import StockDataEngine

class MLModelTrainer:
    """Enhanced ML Model Training with Auto-Training Capabilities"""
    
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
                st.info(f"🚀 Training {analysis_type.replace('_', ' ')} model for {symbol}...")
            
            # Determine prediction horizon
            horizon = 1 if analysis_type == 'next_day' else 5
            
            # Prepare training data
            with st.spinner(f"Fetching data for {symbol}...") if show_progress else st.empty():
                features, labels, raw_data = self.data_engine.prepare_training_data(
                    symbol=symbol, 
                    period=period, 
                    prediction_horizon=horizon
                )
            
            if features is None or len(features) < 50:
                error_msg = f"❌ Insufficient data for {symbol} (got {len(features) if features is not None else 0} samples, need 50+)"
                if show_progress:
                    st.error(error_msg)
                return False, 0.0
            
            if show_progress:
                st.success(f"✅ Data prepared: {len(features)} samples with {len(features.columns)} features")
            
            # Prepare data for training
            X = features.values
            y = labels.values
            
            # Scale features and train model
            with st.spinner(f"Training ML model for {symbol}...") if show_progress else st.empty():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                accuracies = []
                best_model = None
                best_accuracy = 0
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Train model (optimized for speed)
                    model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=8,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=1
                    )
                    
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    accuracies.append(accuracy)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                    
                    if show_progress:
                        st.text(f"  Fold {fold + 1}: {accuracy:.3f} accuracy")
            
            # Save best model and scaler
            model_path = os.path.join(self.models_dir, f"{symbol}_{analysis_type}")
            joblib.dump(best_model, f"{model_path}_model.pkl")
            joblib.dump(scaler, f"{model_path}_scaler.pkl")
            
            avg_accuracy = np.mean(accuracies)
            success_msg = f"✅ {symbol} model trained: {avg_accuracy:.3f} average accuracy"
            
            if show_progress:
                st.success(success_msg)
            
            return True, avg_accuracy
            
        except Exception as e:
            error_msg = f"❌ Training failed for {symbol}: {str(e)}"
            if show_progress:
                st.error(error_msg)
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
    
    def predict_stock_with_auto_training(self, symbol, analysis_type='next_day'):
        """Enhanced prediction with automatic training if model doesn't exist"""
        try:
            # Check if model exists, if not, train it
            if not self.model_exists(symbol, analysis_type):
                st.warning(f"🔄 Model for {symbol} ({analysis_type.replace('_', ' ')}) not found. Training automatically...")
                
                with st.expander(f"Training Details for {symbol}", expanded=True):
                    success, accuracy = self.train_stock_model(symbol, analysis_type, show_progress=True)
                    
                    if not success:
                        st.error(f"Failed to train model for {symbol}. Please check data availability.")
                        return None
                    
                    st.info(f"Model trained with {accuracy:.1%} accuracy")
            
            # Load model
            model, scaler = self.load_model(symbol, analysis_type)
            if model is None or scaler is None:
                st.error(f"Failed to load model for {symbol} even after training attempt.")
                return None
            
            # Get latest data and features
            horizon = 1 if analysis_type == 'next_day' else 5
            features, _, raw_data = self.data_engine.prepare_training_data(
                symbol=symbol, 
                period='6mo', 
                prediction_horizon=horizon
            )
            
            if features is None or len(features) == 0:
                st.error(f"Unable to fetch current data for {symbol}")
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
            st.error(f"Prediction error for {symbol}: {str(e)}")
            return None
    
    def get_trained_models_status(self):
        """Get status of all trained models"""
        if not os.path.exists(self.models_dir):
            return {}
        
        models = {}
        for file in os.listdir(self.models_dir):
            if file.endswith('_model.pkl'):
                parts = file.replace('_model.pkl', '').split('_')
                if len(parts) >= 2:
                    symbol = '_'.join(parts[:-1])
                    analysis_type = parts[-1]
                    
                    if symbol not in models:
                        models[symbol] = []
                    models[symbol].append(analysis_type)
        
        return models

if __name__ == "__main__":
    # Test the model trainer
    trainer = MLModelTrainer()
    success, accuracy = trainer.train_stock_model('RELIANCE', 'next_day')
    print(f"Training success: {success}, Accuracy: {accuracy}")
