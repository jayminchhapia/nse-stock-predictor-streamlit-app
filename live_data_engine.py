import yfinance as yf
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Callable

class LiveDataEngine:
    """Yahoo Finance Live Data Engine - FIXED for Market Hours"""
    
    def __init__(self, refresh_interval=30):
        self.refresh_interval = refresh_interval
        self.is_streaming = False
        self.subscribers = {}
        self.latest_prices = {}
        self.streaming_thread = None
        self.data_cache = {}
        self.last_fetch_time = {}
    
    def add_subscriber(self, symbol: str, callback: Callable):
        """Add a callback function for price updates"""
        symbol = symbol.upper()
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
    
    def remove_subscriber(self, symbol: str, callback: Callable):
        """Remove a callback function"""
        symbol = symbol.upper()
        if symbol in self.subscribers:
            try:
                self.subscribers[symbol].remove(callback)
                if not self.subscribers[symbol]:
                    del self.subscribers[symbol]
            except ValueError:
                pass
    
    def _is_market_open(self) -> bool:
        """Check if NSE market is currently open"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Skip weekends
        if weekday >= 5:  # Saturday=5, Sunday=6
            return False
        
        # NSE trading hours: 9:15 AM to 3:30 PM IST
        if hour == 9 and minute >= 15:
            return True
        elif 10 <= hour <= 14:
            return True
        elif hour == 15 and minute <= 30:
            return True
        else:
            return False
    
    def _fetch_last_close_price(self, symbol: str) -> Dict:
        """Fetch last available close price when market is closed"""
        try:
            symbol_yf = symbol + '.NS' if not symbol.endswith('.NS') else symbol
            ticker = yf.Ticker(symbol_yf)
            
            # Get last 2 days of data
            hist = ticker.history(period='2d', interval='1d')
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns and not pd.isna(hist['Volume'].iloc[-1]) else 0
                
                return {
                    'symbol': symbol.replace('.NS', ''),
                    'price': current_price,
                    'previous_close': prev_close,
                    'change': current_price - prev_close,
                    'change_percent': ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0,
                    'volume': volume,
                    'open': float(hist['Open'].iloc[-1]) if 'Open' in hist.columns else current_price,
                    'high': float(hist['High'].iloc[-1]) if 'High' in hist.columns else current_price,
                    'low': float(hist['Low'].iloc[-1]) if 'Low' in hist.columns else current_price,
                    'timestamp': datetime.now(),
                    'market_state': 'CLOSED',
                    'data_source': 'yahoo_close',
                    'delay_minutes': 0,
                    'note': '🕐 Market Closed - Showing last close price'
                }
        except Exception as e:
            print(f"Error fetching last close for {symbol}: {e}")
            
        return None
    
    def get_live_price(self, symbol: str) -> Dict:
        """Get live price with market hours handling - FIXED"""
        try:
            symbol = symbol.upper()
            current_time = datetime.now()
            
            # Check cache first (5 second cache for closed market, 30 second for open)
            cache_duration = 5 if not self._is_market_open() else 30
            
            if symbol in self.last_fetch_time:
                time_diff = (current_time - self.last_fetch_time[symbol]).seconds
                if time_diff < cache_duration and symbol in self.data_cache:
                    return self.data_cache[symbol]
            
            # If market is closed, fetch last close price
            if not self._is_market_open():
                price_data = self._fetch_last_close_price(symbol)
                if price_data:
                    self.data_cache[symbol] = price_data
                    self.last_fetch_time[symbol] = current_time
                    return price_data
            
            # Market is open - try to fetch live data
            symbol_yf = symbol + '.NS' if not symbol.endswith('.NS') else symbol
            ticker = yf.Ticker(symbol_yf)
            
            # Method 1: Try recent intraday data
            try:
                hist = ticker.history(period='1d', interval='1m')
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns and not pd.isna(hist['Volume'].iloc[-1]) else 0
                    
                    # Get previous close from info
                    try:
                        info = ticker.info
                        prev_close = float(info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price))
                    except:
                        prev_close = float(hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
                    
                    # OHLC from today's data
                    today_data = hist.tail(60)  # Last hour
                    if not today_data.empty:
                        day_open = float(today_data['Open'].iloc[0])
                        day_high = float(today_data['High'].max())
                        day_low = float(today_data['Low'].min())
                    else:
                        day_open = day_high = day_low = current_price
                    
                    price_data = {
                        'symbol': symbol.replace('.NS', ''),
                        'price': current_price,
                        'previous_close': prev_close,
                        'change': current_price - prev_close,
                        'change_percent': ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0,
                        'volume': volume,
                        'open': day_open,
                        'high': day_high,
                        'low': day_low,
                        'timestamp': current_time,
                        'market_state': 'OPEN',
                        'data_source': 'yahoo_live',
                        'delay_minutes': 15,
                        'note': '📈 Live Market Data (~15 min delay)'
                    }
                    
                    # Cache the result
                    self.data_cache[symbol] = price_data
                    self.last_fetch_time[symbol] = current_time
                    
                    return price_data
            
            except Exception as e:
                print(f"Live data fetch failed for {symbol}: {e}")
            
            # Fallback to last close price
            price_data = self._fetch_last_close_price(symbol)
            if price_data:
                price_data['note'] = '⚠️ Live data unavailable - Showing last close'
                self.data_cache[symbol] = price_data
                self.last_fetch_time[symbol] = current_time
                return price_data
            
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            
        # Ultimate fallback - return empty data structure
        return {
            'symbol': symbol,
            'price': 0.0,
            'previous_close': 0.0,
            'change': 0.0,
            'change_percent': 0.0,
            'volume': 0,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'timestamp': datetime.now(),
            'market_state': 'ERROR',
            'data_source': 'error',
            'delay_minutes': 0,
            'note': '❌ Unable to fetch data'
        }
    
    def start_streaming(self, symbols: List[str]):
        """Start streaming with proper error handling"""
        if self.is_streaming:
            self.stop_streaming()
            
        self.is_streaming = True
        symbols = [s.upper() for s in symbols]
        
        def stream_worker():
            cycle_count = 0
            while self.is_streaming:
                cycle_count += 1
                print(f"📡 Streaming cycle {cycle_count} for {len(symbols)} symbols...")
                
                for symbol in symbols:
                    if not self.is_streaming:
                        break
                        
                    if symbol in self.subscribers:
                        try:
                            price_data = self.get_live_price(symbol)
                            if price_data:
                                self.latest_prices[symbol] = price_data
                                
                                # Notify subscribers
                                for callback in self.subscribers[symbol]:
                                    try:
                                        callback(price_data)
                                    except Exception as e:
                                        print(f"Callback error for {symbol}: {e}")
                        except Exception as e:
                            print(f"Streaming error for {symbol}: {e}")
                
                # Wait before next refresh
                if self.is_streaming:
                    time.sleep(self.refresh_interval)
        
        self.streaming_thread = threading.Thread(target=stream_worker, daemon=True)
        self.streaming_thread.start()
        print(f"✅ Started streaming for {len(symbols)} symbols")
    
    def stop_streaming(self):
        """Stop streaming safely"""
        self.is_streaming = False
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2)
        print("🛑 Streaming stopped")
    
    def get_latest_price(self, symbol: str) -> Dict:
        """Get latest cached price or fetch fresh"""
        symbol = symbol.upper()
        return self.latest_prices.get(symbol) or self.get_live_price(symbol)
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices for multiple symbols"""
        results = {}
        for symbol in symbols:
            try:
                price_data = self.get_live_price(symbol)
                if price_data and price_data.get('price', 0) > 0:
                    results[symbol.upper()] = price_data
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        return results
    
    def get_market_summary(self, symbols: List[str]) -> Dict:
        """Get market summary with error handling"""
        prices = self.get_multiple_prices(symbols)
        
        if not prices:
            return {
                'total_stocks': 0,
                'gainers': 0,
                'losers': 0,
                'unchanged': 0,
                'avg_change_percent': 0,
                'market_state': 'UNKNOWN',
                'last_updated': datetime.now()
            }
        
        total_gainers = sum(1 for p in prices.values() if p.get('change', 0) > 0)
        total_losers = sum(1 for p in prices.values() if p.get('change', 0) < 0)
        total_unchanged = len(prices) - total_gainers - total_losers
        
        changes = [p.get('change_percent', 0) for p in prices.values()]
        avg_change = np.mean(changes) if changes else 0
        
        return {
            'total_stocks': len(prices),
            'gainers': total_gainers,
            'losers': total_losers,
            'unchanged': total_unchanged,
            'avg_change_percent': avg_change,
            'market_state': list(prices.values())[0]['market_state'] if prices else 'UNKNOWN',
            'last_updated': datetime.now()
        }

# Global instance
live_engine = LiveDataEngine(refresh_interval=30)