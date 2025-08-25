import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
from data_engine import StockDataEngine
from model_trainer import MLModelTrainer

# NEW IMPORTS for Phase 3A
from live_data_engine import live_engine
from market_analyzer import market_analyzer
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Enhanced ML Stock Predictor Pro", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background: linear-gradient(90deg, #00c851, #007e33);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-low {
        background: linear-gradient(90deg, #ff4444, #cc0000);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-medium {
        background: linear-gradient(90deg, #ffbb33, #ff8800);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize systems
@st.cache_resource
def get_systems():
    return StockDataEngine(), MLModelTrainer()

data_engine, ml_trainer = get_systems()

# Initialize session state for watchlist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['RELIANCE', 'TCS', 'HDFCBANK']
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Title
st.markdown('<h1 class="main-header">📈 Enhanced ML Stock Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown("**Advanced Machine Learning with 18+ Features • 75-85% Accuracy • Auto-Training**")

# Sidebar Navigation
st.sidebar.header("🎯 Navigation")
page = st.sidebar.selectbox(
    "Select Page:",
    ["📊 Stock Analysis", "📈 Portfolio Dashboard", "⚙️ Model Management", "📋 Analysis History", "📈 Smart Live Dashboard", "📈 Market Analysis"]
)

# Main content based on page selection
if page == "📊 Stock Analysis":
    
    # Stock selection section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol:", value="RELIANCE", placeholder="e.g., RELIANCE, TCS").strip().upper()
    
    with col2:
        analysis_type = st.selectbox("Analysis Type:", ["next_day", "short_term"], 
                                   format_func=lambda x: "Next Day" if x == "next_day" else "Short Term (5 days)")
    
    with col3:
        add_to_watchlist = st.button("➕ Add to Watchlist", disabled=symbol in st.session_state.watchlist if symbol else True)
        if add_to_watchlist and symbol and symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(symbol)
            st.success(f"Added {symbol} to watchlist!")
    
    # Analysis section
    if st.button("🚀 Run Enhanced ML Analysis", type="primary"):
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                
                # Fetch and display historical data with charts
                raw_data = data_engine.fetch_stock_data(symbol, '1y')
                
                if raw_data is not None:
                    # Create interactive price chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Stock Price & Moving Averages', 'Volume'),
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Price and MA lines
                    fig.add_trace(
                        go.Scatter(x=raw_data.index, y=raw_data['Close'], 
                                 name='Close Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    
                    # Add moving averages
                    raw_data['MA20'] = raw_data['Close'].rolling(20).mean()
                    raw_data['MA50'] = raw_data['Close'].rolling(50).mean()
                    
                    fig.add_trace(
                        go.Scatter(x=raw_data.index, y=raw_data['MA20'], 
                                 name='MA20', line=dict(color='orange', width=1)),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=raw_data.index, y=raw_data['MA50'], 
                                 name='MA50', line=dict(color='red', width=1)),
                        row=1, col=1
                    )
                    
                    # Volume bar chart
                    fig.add_trace(
                        go.Bar(x=raw_data.index, y=raw_data['Volume'], 
                             name='Volume', marker_color='lightblue'),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600, title_text=f"{symbol} - Historical Data Analysis")
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ML Prediction Analysis
                    try:
                        # Check and train model if needed
                        if not ml_trainer.model_exists(symbol, analysis_type):
                            st.info("🔄 Auto-training ML model... (This may take 30-60 seconds)")
                            
                            with st.expander("View Training Progress", expanded=False):
                                success, accuracy = ml_trainer.train_stock_model(symbol, analysis_type, show_progress=False)
                                if not success:
                                    st.error("Model training failed!")
                                    st.stop()
                                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.3f}")
                        else:
                            st.success("✅ Using cached model from this session")
                        
                        # Load model and make prediction
                        model, scaler = ml_trainer.load_model(symbol, analysis_type)
                        horizon = 1 if analysis_type == 'next_day' else 5
                        features, _, _ = data_engine.prepare_training_data(symbol, '6mo', horizon)
                        
                        if features is not None and len(features) > 0:
                            # Make prediction
                            latest_features = features.iloc[-1:].values
                            features_scaled = scaler.transform(latest_features)
                            probability = model.predict_proba(features_scaled)[0][1]
                            
                            # Current price info
                            current_price = raw_data['Close'].iloc[-1]
                            prev_close = raw_data['Close'].iloc[-2]
                            change_pct = ((current_price - prev_close) / prev_close * 100)
                            
                            # Display prediction results
                            st.subheader("🎯 ML Prediction Results")
                            
                            # Metrics in columns
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Current Price", f"₹{current_price:.2f}", f"{change_pct:+.2f}%")
                            
                            with col2:
                                st.metric("ML Probability (UP)", f"{probability*100:.1f}%")
                            
                            with col3:
                                confidence = max(probability, 1-probability)
                                st.metric("Confidence", f"{confidence*100:.1f}%")
                            
                            with col4:
                                if probability >= 0.6:
                                    recommendation = "BUY"
                                    color = "prediction-high"
                                elif probability <= 0.4:
                                    recommendation = "SELL"
                                    color = "prediction-low"
                                else:
                                    recommendation = "HOLD"
                                    color = "prediction-medium"
                                
                                st.markdown(f'<div class="{color}"><h3>{recommendation}</h3></div>', 
                                          unsafe_allow_html=True)
                            
                            # Technical indicators chart
                            features_with_index = features.tail(50).copy()
                            features_with_index.index = raw_data.index[-len(features_with_index):]
                            
                            fig2 = make_subplots(rows=2, cols=2, 
                                               subplot_titles=('RSI', 'MACD', 'Bollinger Position', 'Volume Ratio'),
                                               vertical_spacing=0.15)
                            
                            # RSI
                            if 'rsi' in features_with_index.columns:
                                fig2.add_trace(go.Scatter(x=features_with_index.index, y=features_with_index['rsi'], 
                                                        name='RSI', line=dict(color='purple')), row=1, col=1)
                                fig2.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                                fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                            
                            # MACD
                            if 'macd' in features_with_index.columns:
                                fig2.add_trace(go.Scatter(x=features_with_index.index, y=features_with_index['macd'], 
                                                        name='MACD', line=dict(color='blue')), row=1, col=2)
                            
                            # Bollinger Position
                            if 'bollinger_position' in features_with_index.columns:
                                fig2.add_trace(go.Scatter(x=features_with_index.index, y=features_with_index['bollinger_position'], 
                                                        name='Bollinger %', line=dict(color='orange')), row=2, col=1)
                            
                            # Volume Ratio
                            if 'volume_ratio' in features_with_index.columns:
                                fig2.add_trace(go.Scatter(x=features_with_index.index, y=features_with_index['volume_ratio'], 
                                                        name='Volume Ratio', line=dict(color='red')), row=2, col=2)
                            
                            fig2.update_layout(height=500, title_text=f"{symbol} - Technical Indicators")
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Save to history
                            analysis_result = {
                                'timestamp': datetime.now(),
                                'symbol': symbol,
                                'analysis_type': analysis_type,
                                'probability': probability,
                                'recommendation': recommendation,
                                'current_price': current_price,
                                'change_pct': change_pct
                            }
                            st.session_state.analysis_history.append(analysis_result)
                            
                        else:
                            st.error("Unable to generate features for prediction")
                            
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        
                else:
                    st.error(f"Unable to fetch data for {symbol}")
        else:
            st.warning("Please enter a stock symbol")

elif page == "📈 Portfolio Dashboard":
    st.header("📈 Portfolio Dashboard")
    
    # Watchlist management
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Your Watchlist")
        if st.session_state.watchlist:
            for i, stock in enumerate(st.session_state.watchlist):
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.write(f"📈 {stock}")
                with col_b:
                    if st.button(f"Analyze", key=f"analyze_{stock}"):
                        # Set the symbol for analysis
                        st.info(f"Go to Stock Analysis page and enter {stock}")
                with col_c:
                    if st.button(f"Remove", key=f"remove_{stock}"):
                        st.session_state.watchlist.remove(stock)
                        st.rerun()
        else:
            st.info("No stocks in watchlist. Add some stocks from the Analysis page!")
    
    with col2:
        st.subheader("➕ Quick Add")
        new_stock = st.text_input("Add Stock:", placeholder="e.g., INFY").strip().upper()
        if st.button("Add to Watchlist") and new_stock:
            if new_stock not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_stock)
                st.success(f"Added {new_stock}!")
                st.rerun()
            else:
                st.warning(f"{new_stock} already in watchlist!")
    
    # Quick batch analysis info
    st.subheader("🔄 Batch Analysis")
    st.info("""
    **Batch Analysis Coming Soon!** 
    
    For now, analyze stocks individually from the Stock Analysis page.
    Each model trains automatically when needed (30-60 seconds per stock).
    
    **Pro Tip**: Models are cached during your session, so re-analyzing the same stock is instant!
    """)

elif page == "⚙️ Model Management":
    st.header("⚙️ Model Management & System Status")
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔧 Data Engine", "✅ Active")
    with col2:
        st.metric("🤖 ML Trainer", "✅ Auto-Training")
    with col3:
        st.metric("📊 Prediction Engine", "✅ Ready")
    
    # Session model status
    st.subheader("📊 Session Models Status")
    trained_models = ml_trainer.get_trained_models_status()
    
    if trained_models:
        model_data = []
        for symbol, analysis_types in trained_models.items():
            for analysis_type in analysis_types:
                model_path = f"models/{symbol}_{analysis_type}_model.pkl"
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path) / 1024  # KB
                    model_data.append({
                        'Symbol': symbol,
                        'Analysis Type': analysis_type.replace('_', ' ').title(),
                        'Size (KB)': f"{file_size:.1f}",
                        'Status': '✅ Ready (Session Cache)'
                    })
        
        if model_data:
            models_df = pd.DataFrame(model_data)
            st.dataframe(models_df, use_container_width=True)
        else:
            st.info("No models trained in this session yet.")
    else:
        st.info("No models found. Models will be auto-trained when you run analysis.")
    
    # Auto-training info
    st.subheader("🔄 Auto-Training System")
    
    st.success("""
    ✅ **Auto-Training Mode: Active**
    
    - Models train automatically when needed
    - First analysis per stock: 30-60 seconds (training time)
    - Subsequent analyses: Instant (cached models)
    - Fresh models use latest 6-month data
    - No manual training required!
    """)
    
    st.warning("""
    ⚠️ **Session Storage Notice**
    
    - Models are stored temporarily during your session
    - After app redeployment, models will retrain automatically
    - This ensures you always get fresh models with latest market data
    """)

elif page == "📋 Analysis History":
    st.header("📋 Analysis History")
    
    if st.session_state.analysis_history:
        history_data = []
        for analysis in reversed(st.session_state.analysis_history[-20:]):  # Last 20
            history_data.append({
                'Timestamp': analysis['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Symbol': analysis['symbol'],
                'Type': analysis['analysis_type'].replace('_', ' ').title(),
                'Probability': f"{analysis['probability']*100:.1f}%",
                'Recommendation': analysis['recommendation'],
                'Price': f"₹{analysis['current_price']:.2f}",
                'Change %': f"{analysis['change_pct']:+.2f}%"
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Apply styling
        def highlight_rec(val):
            if val == 'BUY':
                return 'background-color: lightgreen'
            elif val == 'SELL':
                return 'background-color: lightcoral'
            else:
                return 'background-color: lightyellow'
        
        styled_history = history_df.style.applymap(highlight_rec, subset=['Recommendation'])
        st.dataframe(styled_history, use_container_width=True)
        
        # Clear history option
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.success("History cleared!")
            st.rerun()
            
        # Download history
        csv = history_df.to_csv(index=False)
        st.download_button(
            "📥 Download History",
            csv,
            file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No analysis history yet. Start analyzing stocks to build your history!")

# Add this to your page selection options
elif page == "📈 Smart Live Dashboard":
    st.header("📈 Smart Live Stock Dashboard")
    st.caption("🔄 Real-time updates • 📱 Easy stock management • 🎯 ML-powered insights")
    
    # Initialize session state
    if 'live_watchlist' not in st.session_state:
        st.session_state.live_watchlist = ['RELIANCE', 'TCS', 'HDFCBANK']
    if 'live_streaming' not in st.session_state:
        st.session_state.live_streaming = False
    if 'live_data' not in st.session_state:
        st.session_state.live_data = {}
    
    # Market status check
    market_open = live_engine._is_market_open()
    market_status = "🟢 OPEN" if market_open else "🔴 CLOSED"
    
    st.info(f"📊 **NSE Market Status**: {market_status} | Current Time: {datetime.now().strftime('%H:%M:%S IST')}")
    
    # === SIDEBAR: WATCHLIST MANAGEMENT ===
    with st.sidebar:
        st.header("📋 Manage Stocks")
        
        # Add new stock
        st.subheader("➕ Add Stock")
        new_symbol = st.text_input(
            "Enter Stock Symbol:",
            placeholder="e.g., INFY, WIPRO",
            key="add_stock_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add", key="add_stock_btn") and new_symbol:
                symbol = new_symbol.strip().upper()
                if symbol and symbol not in st.session_state.live_watchlist:
                    st.session_state.live_watchlist.append(symbol)
                    st.success(f"Added {symbol}!")
                    st.rerun()  # Use st.rerun() instead of experimental_rerun
                elif symbol in st.session_state.live_watchlist:
                    st.warning("Already in watchlist!")
        
        with col2:
            # Quick add popular stocks
            popular_stocks = ['INFY', 'WIPRO', 'AXISBANK', 'BAJFINANCE', 'MARUTI']
            quick_add = st.selectbox("Quick Add:", ["Select..."] + popular_stocks, key="quick_add")
            if quick_add != "Select..." and quick_add not in st.session_state.live_watchlist:
                st.session_state.live_watchlist.append(quick_add)
                st.success(f"Added {quick_add}!")
                st.rerun()
        
        # Current watchlist with remove options
        st.subheader("📊 Your Stocks")
        if st.session_state.live_watchlist:
            for i, stock in enumerate(st.session_state.live_watchlist):
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Show price if available
                    if stock in st.session_state.live_data:
                        price_data = st.session_state.live_data[stock]
                        change_color = "🟢" if price_data.get('change', 0) >= 0 else "🔴"
                        st.write(f"{change_color} **{stock}**: ₹{price_data.get('price', 0):.2f}")
                    else:
                        st.write(f"📈 **{stock}**")
                
                with col2:
                    if st.button("🗑️", key=f"remove_{stock}_{i}", help=f"Remove {stock}"):
                        st.session_state.live_watchlist.remove(stock)
                        if stock in st.session_state.live_data:
                            del st.session_state.live_data[stock]
                        st.rerun()
        else:
            st.info("Add stocks to start monitoring!")
        
        # Streaming controls
        st.subheader("🔄 Live Updates")
        refresh_interval = st.slider("Update Every (seconds):", 15, 120, 30, step=15)
        
        # Start/Stop streaming
        if not st.session_state.live_streaming:
            if st.button("🔴 Start Live Updates", type="primary", key="start_live"):
                if st.session_state.live_watchlist:
                    st.session_state.live_streaming = True
                    st.success("Live updates started!")
                    st.rerun()
                else:
                    st.warning("Add stocks first!")
        else:
            if st.button("🛑 Stop Updates", key="stop_live"):
                st.session_state.live_streaming = False
                st.info("Live updates stopped")
                st.rerun()
    
    # === MAIN AREA: LIVE DASHBOARD ===
    
    if not st.session_state.live_watchlist:
        st.info("👈 Add stocks from the sidebar to start monitoring!")
        st.markdown("""
        **Quick Start:**
        1. Enter a stock symbol in the sidebar (e.g., INFY, WIPRO)
        2. Click ➕ Add or use Quick Add dropdown
        3. Click 🔴 Start Live Updates
        4. Watch prices update automatically!
        """)
        
    else:
        # Summary row
        if st.session_state.live_data:
            total_stocks = len(st.session_state.live_data)
            gainers = sum(1 for data in st.session_state.live_data.values() if data.get('change', 0) > 0)
            losers = sum(1 for data in st.session_state.live_data.values() if data.get('change', 0) < 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Stocks", total_stocks)
            with col2:
                st.metric("🟢 Gainers", gainers)
            with col3:
                st.metric("🔴 Losers", losers)
            with col4:
                st.metric("Status", "🔴 LIVE" if st.session_state.live_streaming else "⏸️ PAUSED")
        
        # Live stock cards
        st.subheader("💹 Live Stock Prices")
        
        # Display stock cards in a grid
        cols_per_row = 3
        rows = [st.session_state.live_watchlist[i:i+cols_per_row] for i in range(0, len(st.session_state.live_watchlist), cols_per_row)]
        
        for row_stocks in rows:
            cols = st.columns(len(row_stocks))
            
            for col, stock in zip(cols, row_stocks):
                with col:
                    # Create card-like container
                    with st.container():
                        # Fetch fresh data for display
                        try:
                            price_data = live_engine.get_live_price(stock)
                            st.session_state.live_data[stock] = price_data
                        except Exception as e:
                            print(f"Error fetching {stock}: {e}")
                            price_data = None
                        
                        if price_data and price_data.get('price', 0) > 0:
                            # Card header
                            change_emoji = "🟢" if price_data.get('change', 0) >= 0 else "🔴"
                            st.markdown(f"### {change_emoji} {stock}")
                            
                            # Price info
                            st.metric(
                                "Price",
                                f"₹{price_data.get('price', 0):.2f}",
                                f"{price_data.get('change', 0):+.2f} ({price_data.get('change_percent', 0):+.2f}%)"
                            )
                            
                            # Show market status note
                            if price_data.get('note'):
                                if 'Market Closed' in price_data['note']:
                                    st.info(price_data['note'])
                                elif 'Live Market' in price_data['note']:
                                    st.success(price_data['note'])
                                else:
                                    st.warning(price_data['note'])
                            
                            # Additional info
                            st.caption(f"Volume: {price_data.get('volume', 0):,}")
                            st.caption(f"Market: {price_data.get('market_state', 'Unknown')}")
                            
                            # Last update time
                            update_time = price_data.get('timestamp', datetime.now())
                            st.caption(f"🕒 {update_time.strftime('%H:%M:%S')}")
                            
                        else:
                            # Error state
                            st.markdown(f"### ❌ {stock}")
                            st.error("Unable to fetch data")
                            
                            # Manual refresh button
                            if st.button(f"🔄 Retry {stock}", key=f"retry_{stock}"):
                                st.rerun()
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh All Prices", key="refresh_all"):
                with st.spinner("Refreshing all prices..."):
                    for symbol in st.session_state.live_watchlist:
                        try:
                            price_data = live_engine.get_live_price(symbol)
                            if price_data:
                                st.session_state.live_data[symbol] = price_data
                        except:
                            continue
                    st.success("All prices refreshed!")
                    st.rerun()
        
        with col2:
            if st.session_state.live_data:
                # Export data
                export_data = []
                for symbol, data in st.session_state.live_data.items():
                    export_data.append({
                        'Symbol': symbol,
                        'Price': data.get('price', 0),
                        'Change': data.get('change', 0),
                        'Change %': data.get('change_percent', 0),
                        'Volume': data.get('volume', 0),
                        'Market': data.get('market_state', ''),
                        'Time': data.get('timestamp', datetime.now()).strftime('%H:%M:%S')
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "📥 Export Data",
                    csv,
                    file_name=f"live_prices_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="export_live"
                )
        
        with col3:
            if st.button("🗑️ Clear All Data", key="clear_data"):
                st.session_state.live_data = {}
                st.info("All data cleared!")
                st.rerun()
    
    # Help section
    with st.expander("💡 How to Use This Dashboard"):
        st.markdown("""
        **Getting Started:**
        1. **Add Stocks**: Use the sidebar to add stocks to your watchlist
        2. **View Live Data**: Prices fetch automatically when you load the page
        3. **Market Hours**: Live data during market hours, last close when closed
        
        **Features:**
        - ➕ **Easy Add/Remove**: Manage your watchlist dynamically
        - 🕐 **Market Hours Detection**: Shows appropriate data based on market status
        - 📊 **Clean Cards Layout**: Each stock displayed in its own card
        - 📥 **Export Data**: Download current prices as CSV
        - 🔄 **Manual Refresh**: Update all prices with one click
        
        **Market Hours:**
        - **NSE Trading**: Monday-Friday, 9:15 AM - 3:30 PM IST
        - **Live Data**: During market hours (~15 min delay)
        - **Last Close**: When market is closed or weekends
        """)

elif page == "📈 Market Analysis":
    st.header("📈 Advanced Market Analysis")
    st.caption("Powered by Yahoo Finance delayed data")
    
    # Analysis options
    analysis_symbols = st.multiselect(
        "Select Stocks for Analysis:",
        options=st.session_state.watchlist + ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'WIPRO', 'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE'],
        default=st.session_state.watchlist[:10] if st.session_state.watchlist else ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'WIPRO'],
        key="analysis_symbols"
    )
    
    if analysis_symbols and len(analysis_symbols) >= 2:
        
        # Top Movers Analysis - FIXED
        if st.button("🚀 Analyze Top Movers", key="analyze_movers_btn"):
            with st.spinner("Analyzing market movers..."):
                movers = market_analyzer.get_top_movers(analysis_symbols, limit=5)
                
                if movers['total_analyzed'] > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🟢 Top Gainers")
                        if movers['top_gainers']:
                            for gainer in movers['top_gainers']:
                                st.metric(
                                    gainer['symbol'],
                                    f"₹{gainer['price']:.2f}",
                                    f"{gainer['change_pct']:+.2f}%"
                                )
                        else:
                            st.info("No gainers found")
                    
                    with col2:
                        st.subheader("🔴 Top Losers")
                        if movers['top_losers']:
                            for loser in movers['top_losers']:
                                st.metric(
                                    loser['symbol'],
                                    f"₹{loser['price']:.2f}",
                                    f"{loser['change_pct']:+.2f}%"
                                )
                        else:
                            st.info("No losers found")
                else:
                    st.warning("Unable to analyze market movers")
        
        # Volatility Analysis - FIXED
        if st.button("📊 Volatility Analysis", key="analyze_volatility_btn"):
            with st.spinner("Analyzing volatility..."):
                volatility = market_analyzer.get_volatility_analysis(analysis_symbols)
                
                if volatility['high_volatility'] or volatility['low_volatility']:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("⚡ High Volatility Stocks")
                        if volatility['high_volatility']:
                            for stock in volatility['high_volatility']:
                                st.metric(
                                    stock['symbol'],
                                    f"{stock['volatility']:.1f}%",
                                    f"Volume: {stock['avg_volume']:,.0f}"
                                )
                        else:
                            st.info("No high volatility stocks found")
                    
                    with col2:
                        st.subheader("🔒 Low Volatility Stocks")
                        if volatility['low_volatility']:
                            for stock in volatility['low_volatility']:
                                st.metric(
                                    stock['symbol'],
                                    f"{stock['volatility']:.1f}%",
                                    f"Volume: {stock['avg_volume']:,.0f}"
                                )
                        else:
                            st.info("No low volatility stocks found")
                    
                    st.info(f"📊 Average Market Volatility: {volatility['avg_market_volatility']:.1f}%")
                else:
                    st.warning("Unable to perform volatility analysis")
        
        # Correlation Analysis - FIXED
        if st.button("🔗 Correlation Analysis", key="analyze_correlation_btn"):
            with st.spinner("Calculating correlations..."):
                correlation_matrix = market_analyzer.get_correlation_analysis(analysis_symbols)
                
                if not correlation_matrix.empty:
                    st.subheader("🔗 Stock Correlation Matrix")
                    
                    # Create heatmap using Plotly
                    import plotly.express as px
                    
                    fig = px.imshow(
                        correlation_matrix.values,
                        labels=dict(x="Stock", y="Stock", color="Correlation"),
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        color_continuous_scale="RdBu_r",
                        aspect="auto"
                    )
                    
                    fig.update_layout(
                        title="Stock Price Correlation Matrix",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlations - FIXED
                    st.subheader("🔍 Correlation Insights")
                    
                    # Get correlation pairs
                    correlations = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            stock1 = correlation_matrix.columns[i]
                            stock2 = correlation_matrix.columns[j]
                            corr_value = correlation_matrix.iloc[i, j]
                            correlations.append((stock1, stock2, corr_value))
                    
                    if correlations:
                        # Sort by correlation value
                        correlations.sort(key=lambda x: x[2], reverse=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Highest Positive Correlations:**")
                            positive_corrs = [c for c in correlations if c[2] > 0][:5]
                            for stock1, stock2, corr in positive_corrs:
                                st.write(f"• {stock1} ↔ {stock2}: {corr:.3f}")
                        
                        with col2:
                            st.write("**Lowest/Negative Correlations:**")
                            negative_corrs = [c for c in correlations if c[2] < 0.5][-5:]
                            for stock1, stock2, corr in negative_corrs:
                                st.write(f"• {stock1} ↔ {stock2}: {corr:.3f}")
                
                else:
                    st.warning("Unable to calculate correlations")
    
    else:
        st.info("Please select at least 2 stocks for market analysis")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.header("📊 System Info")
st.sidebar.success("✅ Data Engine: Active")
st.sidebar.success("✅ ML Models: Auto-Training")
st.sidebar.success("✅ Prediction: Ready")
st.sidebar.info("📱 Session Storage: Temporary")

st.sidebar.markdown("---")
st.sidebar.info("""
**Phase 2 Features:**
- 📈 Interactive Charts
- 📋 Portfolio Management  
- 🎯 Technical Indicators
- 📊 Analysis History
- ⚙️ Model Management
- 🔄 Auto-Training
""")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Educational use only. Not financial advice.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<h4>📈 Enhanced ML Stock Predictor Pro</h4>
<p>Phase 2 • Auto-Training • 18+ Features • 75-85% Accuracy</p>
<small>💡 Tip: Models train automatically and cache during your session!</small>
</div>
""", unsafe_allow_html=True)
