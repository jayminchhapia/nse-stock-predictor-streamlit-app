import os
from model_trainer import MLModelTrainer

def train_popular_models():
    """Standalone training without Streamlit commands"""
    trainer = MLModelTrainer()
    
    stocks = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'WIPRO', 'MARUTI', 'LT'
    ]
    
    analysis_types = ['next_day', 'short_term']
    
    print("🚀 Starting standalone model training...")
    print(f"📊 Training {len(stocks)} stocks × {len(analysis_types)} models")
    
    results = {}
    successful_models = 0
    
    for i, stock in enumerate(stocks, 1):
        print(f"\n📈 Stock {i}/{len(stocks)}: {stock}")
        
        for j, analysis_type in enumerate(analysis_types, 1):
            print(f"  🔄 Analysis {j}/{len(analysis_types)}: {analysis_type}")
            
            try:
                # Use show_progress=False to avoid Streamlit calls
                success, accuracy = trainer.train_stock_model(
                    stock, 
                    analysis_type, 
                    period='6mo',
                    show_progress=False
                )
                
                if success:
                    successful_models += 1
                    print(f"    ✅ Success: {accuracy:.3f} accuracy")
                    results[f"{stock}_{analysis_type}"] = {
                        'success': True, 
                        'accuracy': accuracy
                    }
                else:
                    print(f"    ❌ Failed to train")
                    results[f"{stock}_{analysis_type}"] = {
                        'success': False, 
                        'accuracy': 0.0
                    }
                    
            except Exception as e:
                print(f"    ❌ Error: {str(e)[:50]}...")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🏆 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful models: {successful_models}/{len(stocks) * len(analysis_types)}")
    print(f"📊 Success rate: {successful_models/(len(stocks) * len(analysis_types))*100:.1f}%")
    
    # Check models directory
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        print(f"📁 Model files created: {len(model_files)}")
        
        if model_files:
            print("📋 Created model files:")
            for file in sorted(model_files):
                file_size = os.path.getsize(os.path.join(models_dir, file)) / 1024
                print(f"   {file} ({file_size:.1f} KB)")
    
    return results

if __name__ == "__main__":
    results = train_popular_models()
    print("\n🎯 Standalone training completed!")
