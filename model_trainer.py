import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from data_engine import StockDataEngine

class MLModelTrainer:
    """ML Model Training without Streamlit dependencies"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.data_engine = StockDataEngine()
        os.makedirs(models_dir, exist_ok=True)
    
    def model_exists(self, symbol, analysis_type='next_day'):
        """Check if trained model exists"""
        model_path = os.path.join(self.models_dir, f"{symbol}_{analysis_type}")
        model_file = f"{model_path}_model.pkl"
        scaler_file = f"{model_path}_scaler.pkl"
        return os.path.exists(model_file) and os.path.exists(scaler_file)
    
    def train_stock_model(self, symbol, analysis_type='next_day', period='1y', show_progress=False):
        """Train model without Streamlit dependencies"""
        try:
            if show_progress:
                print(f"🚀 Training {analysis_type.replace('_', ' ')} model for {symbol}...")
            
            # Determine prediction horizon
            horizon = 1 if analysis_type == 'next_day' else 5
            
            # Prepare training data
            print(f"  📊 Fetching data for {symbol}...")
            features, labels, raw_data = self.data_engine.prepare_training_data(
                symbol=symbol, 
                period=period, 
                prediction_horizon=horizon
            )
            
            if features is None or len(features) < 50:
                print(f"  ❌ Insufficient data for {symbol} (got {len(features) if features is not None else 0} samples)")
                return False, 0.0
            
            print(f"  ✅ Data prepared: {len(features)} samples with {len(features.columns)} features")
            
            # Prepare data for training
            X = features.values
            y = labels.values
            
            # Scale features and train model
            print(f"  🔧 Training ML model...")
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
                
                # Train model
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
                    print(f"    Fold {fold + 1}: {accuracy:.3f} accuracy")
            
            # Save best model and scaler
            model_path = os.path.join(self.models_dir, f"{symbol}_{analysis_type}")
            joblib.dump(best_model, f"{model_path}_model.pkl")
            joblib.dump(scaler, f"{model_path}_scaler.pkl")
            
            avg_accuracy = np.mean(accuracies)
            print(f"  ✅ {symbol} model saved: {avg_accuracy:.3f} average accuracy")
            
            return True, avg_accuracy
            
        except Exception as e:
            print(f"  ❌ Training failed for {symbol}: {str(e)}")
            return False, 0.0
    
    def load_model(self, symbol, analysis_type='next_day'):
        """Load trained model and scaler"""
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

if __name__ == "__main__":
    # Test the model trainer
    trainer = MLModelTrainer()
    success, accuracy = trainer.train_stock_model('RELIANCE', 'next_day')
    print(f"Training success: {success}, Accuracy: {accuracy}")
