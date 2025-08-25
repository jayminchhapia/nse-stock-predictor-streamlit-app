import yfinance as yf
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Callable

class LiveDataEngine:
    """Yahoo Finance Live Data Engine with Smart Refresh"""
    
    def __init__(self, refresh_interval=30):
        self.refresh_interval = refresh_interval  # seconds
        self.is_streaming = False
        self.subscribers = {}  # symbol -> list of callback functions
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
    
    def get_live_price(self, symbol: str) -> Dict:
        """Get current price for NSE symbol with caching"""
        try:
            symbol = symbol.upper()
            current_time = datetime.now()
            
            # Check cache (refresh every 10 seconds for individual requests)
            if symbol in self.last_fetch_time:
                time_diff = (current_time - self.last_fetch_time[symbol]).seconds
                if time_diff < 10 and symbol in self.data_cache:
                    return self.data_cache[symbol]
            
            # Fetch fresh data
            symbol_yf = symbol + '.NS' if not symbol.endswith('.NS') else symbol
            ticker = yf.Ticker(symbol_yf)
            
            # Get latest data using multiple methods for reliability
            try:
                # Method 1: Try recent history (most reliable)
                hist = ticker.history(period='2d', interval='1m')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                    
                    # Get previous close from info or yesterday's data
                    try:
                        info = ticker.info
                        prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
                    except:
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    
                    # Calculate OHLC from recent data
                    recent_data = hist.tail(50)  # Last 50 minutes
                    if not recent_data.empty:
                        day_open = recent_data['Open'].iloc[0]
                        day_high = recent_data['High'].max()
                        day_low = recent_data['Low'].min()
                    else:
                        day_open = day_high = day_low = current_price
                    
                    price_data = {
                        'symbol': symbol.replace('.NS', ''),
                        'price': float(current_price),
                        'previous_close': float(prev_close),
                        'change': float(current_price - prev_close),
                        'change_percent': float(((current_price - prev_close) / prev_close * 100)) if prev_close != 0 else 0,
                        'volume': int(volume) if not pd.isna(volume) else 0,
                        'open': float(day_open),
                        'high': float(day_high),
                        'low': float(day_low),
                        'timestamp': current_time,
                        'market_state': self._get_market_state(),
                        'data_source': 'yahoo_finance',
                        'delay_minutes': 15  # Yahoo Finance standard delay
                    }
                    
                    # Cache the result
                    self.data_cache[symbol] = price_data
                    self.last_fetch_time[symbol] = current_time
                    
                    return price_data
            
            except Exception as e:
                print(f"Method 1 failed for {symbol}: {e}")
                
                # Method 2: Fallback to basic ticker info
                try:
                    info = ticker.info
                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                    
                    if current_price and current_price > 0:
                        prev_close = info.get('previousClose', current_price)
                        
                        price_data = {
                            'symbol': symbol.replace('.NS', ''),
                            'price': float(current_price),
                            'previous_close': float(prev_close),
                            'change': float(current_price - prev_close),
                            'change_percent': float(((current_price - prev_close) / prev_close * 100)) if prev_close != 0 else 0,
                            'volume': int(info.get('volume', 0)),
                            'open': float(info.get('open', current_price)),
                            'high': float(info.get('dayHigh', current_price)),
                            'low': float(info.get('dayLow', current_price)),
                            'timestamp': current_time,
                            'market_state': self._get_market_state(),
                            'data_source': 'yahoo_info',
                            'delay_minutes': 15
                        }
                        
                        self.data_cache[symbol] = price_data
                        self.last_fetch_time[symbol] = current_time
                        
                        return price_data
                
                except Exception as e2:
                    print(f"Method 2 also failed for {symbol}: {e2}")
            
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            
        return None
    
    def _get_market_state(self) -> str:
        """Determine if NSE market is open (simplified)"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # NSE trading hours: 9:15 AM to 3:30 PM IST (simplified)
        if hour == 9 and minute >= 15:
            return "OPEN"
        elif 10 <= hour <= 14:
            return "OPEN"
        elif hour == 15 and minute <= 30:
            return "OPEN"
        else:
            return "CLOSED"
    
    def start_streaming(self, symbols: List[str]):
        """Start streaming price updates for given symbols"""
        if self.is_streaming:
            return
            
        self.is_streaming = True
        symbols = [s.upper() for s in symbols]
        
        def stream_worker():
            cycle_count = 0
            while self.is_streaming:
                cycle_count += 1
                print(f"📡 Streaming cycle {cycle_count} for {len(symbols)} symbols...")
                
                for symbol in symbols:
                    if not self.is_streaming:  # Check if stopped during loop
                        break
                        
                    if symbol in self.subscribers:
                        price_data = self.get_live_price(symbol)
                        if price_data:
                            self.latest_prices[symbol] = price_data
                            
                            # Notify all subscribers
                            for callback in self.subscribers[symbol]:
                                try:
                                    callback(price_data)
                                except Exception as e:
                                    print(f"Callback error for {symbol}: {e}")
                
                # Wait before next refresh
                if self.is_streaming:
                    time.sleep(self.refresh_interval)
        
        self.streaming_thread = threading.Thread(target=stream_worker, daemon=True)
        self.streaming_thread.start()
        print(f"✅ Started live streaming for {len(symbols)} symbols (refresh: {self.refresh_interval}s)")
    
    def stop_streaming(self):
        """Stop streaming"""
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2)
        print("🛑 Live streaming stopped")
    
    def get_latest_price(self, symbol: str) -> Dict:
        """Get cached latest price"""
        symbol = symbol.upper()
        return self.latest_prices.get(symbol)
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get prices for multiple symbols efficiently"""
        results = {}
        for symbol in symbols:
            price_data = self.get_live_price(symbol)
            if price_data:
                results[symbol.upper()] = price_data
        return results
    
    def get_market_summary(self, symbols: List[str]) -> Dict:
        """Get market summary for given symbols"""
        prices = self.get_multiple_prices(symbols)
        
        if not prices:
            return {}
        
        total_gainers = sum(1 for p in prices.values() if p['change'] > 0)
        total_losers = sum(1 for p in prices.values() if p['change'] < 0)
        total_unchanged = len(prices) - total_gainers - total_losers
        
        avg_change = np.mean([p['change_percent'] for p in prices.values()])
        
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
