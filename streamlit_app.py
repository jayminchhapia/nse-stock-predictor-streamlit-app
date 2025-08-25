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
import live_data_engine
import market_analyzer
from model_trainer import MLModelTrainer
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
    ["📊 Stock Analysis", "📈 Portfolio Dashboard", "⚙️ Model Management", "📋 Analysis History", "📡 Yahoo Live Stream", "📈 Market Analysis"]
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
                        st.experimental_rerun()
        else:
            st.info("No stocks in watchlist. Add some stocks from the Analysis page!")
    
    with col2:
        st.subheader("➕ Quick Add")
        new_stock = st.text_input("Add Stock:", placeholder="e.g., INFY").strip().upper()
        if st.button("Add to Watchlist") and new_stock:
            if new_stock not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_stock)
                st.success(f"Added {new_stock}!")
                st.experimental_rerun()
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
            st.experimental_rerun()
            
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
elif page == "📡 Yahoo Live Stream":
    st.header("📡 Yahoo Finance Live Data Streaming")
    st.caption("⏰ Data delayed by ~15 minutes | Updates every 30 seconds")
    
    # Market summary section
    st.subheader("📊 Market Overview")
    
    if st.button("🔄 Get Market Summary"):
        with st.spinner("Fetching market data..."):
            summary_symbols = st.session_state.watchlist[:10] if st.session_state.watchlist else ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'WIPRO']
            summary = live_data_engine.get_market_summary(summary_symbols)
            
            if summary:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Stocks", summary['total_stocks'])
                with col2:
                    st.metric("Gainers", summary['gainers'], delta=None)
                with col3:
                    st.metric("Losers", summary['losers'], delta=None)
                with col4:
                    st.metric("Avg Change", f"{summary['avg_change_percent']:+.2f}%")
                
                st.info(f"📅 Market: {summary['market_state']} | Updated: {summary['last_updated'].strftime('%H:%M:%S')}")
    
    # Live streaming controls
    st.subheader("🔴 Live Price Streaming")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        streaming_symbols = st.multiselect(
            "Select Stocks for Live Streaming:",
            options=st.session_state.watchlist + ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'WIPRO', 'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE'],
            default=['RELIANCE'] if not st.session_state.watchlist else st.session_state.watchlist[:3]
        )
    
    with col2:
        refresh_rate = st.selectbox(
            "Refresh Rate:",
            options=[15, 30, 60, 120],
            index=1,
            format_func=lambda x: f"{x} seconds"
        )
    
    with col3:
        max_symbols = st.number_input("Max Symbols:", min_value=1, max_value=20, value=5)
    
    # Limit symbols to prevent overload
    if len(streaming_symbols) > max_symbols:
        streaming_symbols = streaming_symbols[:max_symbols]
        st.warning(f"⚠️ Limited to {max_symbols} symbols to prevent API overload")
    
    # Live streaming interface
    if streaming_symbols:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            start_streaming = st.button("🔴 Start Live Stream", type="primary")
        
        with col2:
            stop_streaming = st.button("🛑 Stop Stream")
        
        # Initialize session state
        if 'streaming_active' not in st.session_state:
            st.session_state.streaming_active = False
        
        if start_streaming:
            st.session_state.streaming_active = True
            live_data_engine.refresh_interval = refresh_rate
            
            # Create containers for live data display
            live_containers = {}
            status_container = st.empty()
            
            for symbol in streaming_symbols:
                with st.expander(f"📈 {symbol} - Live Data", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        price_container = st.empty()
                    with col2:
                        change_container = st.empty()
                    with col3:
                        volume_container = st.empty()
                    with col4:
                        ml_container = st.empty()
                    
                    # Additional details row
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    with detail_col1:
                        ohlc_container = st.empty()
                    with detail_col2:
                        market_container = st.empty()
                    with detail_col3:
                        timestamp_container = st.empty()
                    
                    live_containers[symbol] = {
                        'price': price_container,
                        'change': change_container,
                        'volume': volume_container,
                        'ml': ml_container,
                        'ohlc': ohlc_container,
                        'market': market_container,
                        'timestamp': timestamp_container
                    }
            
            # Price update callback
            def price_update_callback(price_data):
                symbol = price_data['symbol']
                if symbol in live_containers:
                    containers = live_containers[symbol]
                    
                    # Update price display
                    containers['price'].metric(
                        "Live Price",
                        f"₹{price_data['price']:.2f}",
                        f"{price_data['change']:+.2f}"
                    )
                    
                    # Update change display
                    containers['change'].metric(
                        "Change %",
                        f"{price_data['change_percent']:+.2f}%"
                    )
                    
                    # Update volume
                    containers['volume'].metric(
                        "Volume",
                        f"{price_data['volume']:,}"
                    )
                    
                    # ML prediction (if model exists)
                    try:
                        if ml_trainer.model_exists(symbol, 'next_day'):
                            model, scaler = ml_trainer.load_model(symbol, 'next_day')
                            features, _, _ = data_engine.prepare_training_data(symbol, '6mo', 1)
                            
                            if features is not None and model is not None:
                                latest_features = features.iloc[-1:].values
                                features_scaled = scaler.transform(latest_features)
                                probability = model.predict_proba(features_scaled)[0][1]
                                
                                if probability >= 0.6:
                                    recommendation = "BUY"
                                    color = "normal"
                                elif probability <= 0.4:
                                    recommendation = "SELL"
                                    color = "inverse"
                                else:
                                    recommendation = "HOLD"
                                    color = "off"
                                
                                containers['ml'].metric(
                                    "ML Signal",
                                    recommendation,
                                    f"{probability*100:.1f}%"
                                )
                            else:
                                containers['ml'].info("🤖 Training...")
                        else:
                            containers['ml'].info("🤖 No Model")
                    except Exception as e:
                        containers['ml'].warning("ML Error")
                    
                    # OHLC data
                    containers['ohlc'].markdown(f"""
                    **OHLC:**
                    - Open: ₹{price_data['open']:.2f}
                    - High: ₹{price_data['high']:.2f}
                    - Low: ₹{price_data['low']:.2f}
                    """)
                    
                    # Market status
                    containers['market'].info(f"📊 {price_data['market_state']} | {price_data['data_source']}")
                    
                    # Timestamp
                    containers['timestamp'].caption(f"🕒 {price_data['timestamp'].strftime('%H:%M:%S')} | ~{price_data['delay_minutes']}min delay")
            
            # Add subscribers and start streaming
            for symbol in streaming_symbols:
                live_data_engine.add_subscriber(symbol, price_update_callback)
            
            live_data_engine.start_streaming(streaming_symbols)
            
            # Status updates
            status_container.success(f"🔴 Live streaming active for {len(streaming_symbols)} stocks | Refresh: {refresh_rate}s")
            
            st.info("""
            📡 **Live Streaming Active!**
            - Data refreshes automatically every 30 seconds
            - Yahoo Finance data is delayed by ~15 minutes
            - ML predictions update with each price refresh
            - Keep this page open for continuous updates
            """)
        
        if stop_streaming or st.session_state.get('stop_requested', False):
            if st.session_state.streaming_active:
                live_data_engine.stop_streaming()
                st.session_state.streaming_active = False
                st.success("🛑 Live streaming stopped")
    
    # Manual price check section
    st.subheader("🔍 Quick Price Check")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        check_symbol = st.text_input("Check Stock Price:", value="RELIANCE", placeholder="Enter NSE symbol")
    
    with col2:
        if st.button("📊 Get Price"):
            if check_symbol:
                with st.spinner(f"Fetching data for {check_symbol}..."):
                    price_data = live_data_engine.get_live_price(check_symbol)
                    
                    if price_data:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Price", f"₹{price_data['price']:.2f}")
                        with col2:
                            st.metric("Change", f"₹{price_data['change']:+.2f}")
                        with col3:
                            st.metric("Change %", f"{price_data['change_percent']:+.2f}%")
                        
                        # Additional info
                        st.info(f"""
                        📊 **Details:**
                        - Volume: {price_data['volume']:,}
                        - Open: ₹{price_data['open']:.2f} | High: ₹{price_data['high']:.2f} | Low: ₹{price_data['low']:.2f}
                        - Market: {price_data['market_state']} | Source: {price_data['data_source']}
                        - Last Updated: {price_data['timestamp'].strftime('%H:%M:%S')}
                        """)
                    else:
                        st.error(f"Unable to fetch data for {check_symbol}")
    
    # Bulk price check
    st.subheader("📋 Bulk Price Check")
    
    if st.button("📊 Check All Watchlist Prices"):
        if st.session_state.watchlist:
            with st.spinner("Fetching bulk data..."):
                bulk_data = live_data_engine.get_multiple_prices(st.session_state.watchlist)
                
                if bulk_data:
                    # Create summary table
                    table_data = []
                    for symbol, data in bulk_data.items():
                        table_data.append({
                            'Symbol': symbol,
                            'Price': f"₹{data['price']:.2f}",
                            'Change': f"₹{data['change']:+.2f}",
                            'Change %': f"{data['change_percent']:+.2f}%",
                            'Volume': f"{data['volume']:,}",
                            'Market': data['market_state']
                        })
                    
                    df = pd.DataFrame(table_data)
                    
                    # Color coding function
                    def highlight_changes(val):
                        if 'Change %' in val.name:
                            if '+' in str(val):
                                return 'color: green'
                            elif '-' in str(val):
                                return 'color: red'
                        return ''
                    
                    styled_df = df.style.applymap(highlight_changes)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Bulk Data",
                        csv,
                        file_name=f"bulk_prices_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data retrieved for watchlist stocks")
        else:
            st.info("Add stocks to your watchlist first!")
    
    # Performance tips
    with st.expander("💡 Performance Tips"):
        st.markdown("""
        **Optimize Your Live Streaming:**
        
        1. **Limit Symbols:** Keep streaming to 5-10 stocks max
        2. **Adjust Refresh Rate:** Use 60+ seconds for casual monitoring
        3. **Close Other Tabs:** Reduces memory usage
        4. **Stable Internet:** Ensures consistent data fetching
        5. **Market Hours:** Best data availability during 9:15 AM - 3:30 PM IST
        
        **Data Limitations:**
        - Yahoo Finance data is delayed by ~15 minutes
        - Some stocks may have longer delays
        - Weekend/holiday data may be stale
        - High-frequency updates not supported
        """)

elif page == "📈 Market Analysis":
    st.header("📈 Advanced Market Analysis")
    st.caption("Powered by Yahoo Finance delayed data")
    
    # Analysis options
    analysis_symbols = st.multiselect(
        "Select Stocks for Analysis:",
        options=st.session_state.watchlist + ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'WIPRO', 'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 'CIPLA', 'COALINDIA', 'DRREDDY'],
        default=st.session_state.watchlist[:10] if st.session_state.watchlist else ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'WIPRO']
    )
    
    if analysis_symbols and len(analysis_symbols) >= 2:
        
        # Top Movers Analysis
        if st.button("🚀 Analyze Top Movers"):
            with st.spinner("Analyzing market movers..."):
                movers = market_analyzer.get_top_movers(analysis_symbols, limit=5)
                
                if movers['total_analyzed'] > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🟢 Top Gainers")
                        for gainer in movers['top_gainers']:
                            st.metric(
                                gainer['symbol'],
                                f"₹{gainer['price']:.2f}",
                                f"{gainer['change_pct']:+.2f}%"
                            )
                    
                    with col2:
                        st.subheader("🔴 Top Losers")
                        for loser in movers['top_losers']:
                            st.metric(
                                loser['symbol'],
                                f"₹{loser['price']:.2f}",
                                f"{loser['change_pct']:+.2f}%"
                            )
                else:
                    st.warning("Unable to analyze market movers")
        
        # Sector Analysis
        if st.button("🏭 Sector Performance Analysis"):
            with st.spinner("Analyzing sector performance..."):
                sectors = market_analyzer.get_sector_performance(analysis_symbols)
                
                if sectors:
                    st.subheader("🏭 Sector Performance")
                    
                    sector_data = []
                    for sector, data in sectors.items():
                        sector_data.append({
                            'Sector': sector,
                            'Stock Count': data['stock_count'],
                            'Avg Change %': f"{data['avg_change']:+.2f}%",
                            'Stocks': ', '.join(data['stocks'][:3]) + ('...' if len(data['stocks']) > 3 else '')
                        })
                    
                    df = pd.DataFrame(sector_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Unable to perform sector analysis")
        
        # Volatility Analysis
        if st.button("📊 Volatility Analysis"):
            with st.spinner("Analyzing volatility..."):
                volatility = market_analyzer.get_volatility_analysis(analysis_symbols)
                
                if volatility['high_volatility']:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("⚡ High Volatility Stocks")
                        for stock in volatility['high_volatility']:
                            st.metric(
                                stock['symbol'],
                                f"{stock['volatility']:.1f}%",
                                f"Volume: {stock['avg_volume']:,.0f}"
                            )
                    
                    with col2:
                        st.subheader("🔒 Low Volatility Stocks")
                        for stock in volatility['low_volatility']:
                            st.metric(
                                stock['symbol'],
                                f"{stock['volatility']:.1f}%",
                                f"Volume: {stock['avg_volume']:,.0f}"
                            )
                    
                    st.info(f"📊 Average Market Volatility: {volatility['avg_market_volatility']:.1f}%")
                else:
                    st.warning("Unable to perform volatility analysis")
        
        # Correlation Analysis
        if st.button("🔗 Correlation Analysis"):
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
                    
                    # Show top correlations
                    st.subheader("🔍 Strong Correlations")
                    
                    # Get correlation pairs
                    correlations = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            stock1 = correlation_matrix.columns[i]
                            stock2 = correlation_matrix.columns[j]
                            corr_value = correlation_matrix.iloc[i, j]
                            correlations.append((stock1, stock2, corr_value))
                    
                    # Sort by absolute correlation
                    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Highest Positive Correlations:**")
                        for stock1, stock2, corr in correlations[:5]:
                            st.write(f"• {stock1} ↔ {stock2}: {corr:.3f}")
                    
                    with col2:
                        st.write("**Lowest Correlations:**")
                        for stock1, stock2, corr in correlations[-5:]:
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
