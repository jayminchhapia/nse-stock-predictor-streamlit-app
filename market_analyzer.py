import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List
import streamlit as st

class MarketAnalyzer:
    """Advanced market analysis using Yahoo Finance data - FIXED VERSION"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def get_sector_performance(self, symbols: List[str]) -> Dict:
        """Analyze sector performance for given symbols - FIXED"""
        try:
            sector_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    info = ticker.info
                    sector = info.get('sector', 'Unknown')
                    
                    # Get recent performance
                    hist = ticker.history(period='5d')
                    if len(hist) >= 2:
                        recent_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                        
                        if sector not in sector_data:
                            sector_data[sector] = {
                                'stocks': [],
                                'changes': [],
                                'avg_change': 0
                            }
                        
                        sector_data[sector]['stocks'].append(symbol)
                        sector_data[sector]['changes'].append(recent_change)
                
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Calculate sector averages
            for sector in sector_data:
                changes = sector_data[sector]['changes']
                sector_data[sector]['avg_change'] = np.mean(changes) if changes else 0
                sector_data[sector]['stock_count'] = len(sector_data[sector]['stocks'])
            
            return sector_data
            
        except Exception as e:
            print(f"Sector analysis error: {e}")
            return {}
    
    def get_top_movers(self, symbols: List[str], limit: int = 5) -> Dict:
        """Get top gainers and losers - FIXED"""
        try:
            movers_data = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    hist = ticker.history(period='2d')
                    
                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2]
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        movers_data.append({
                            'symbol': symbol,
                            'price': float(current_price),
                            'change_pct': float(change_pct),
                            'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns and not pd.isna(hist['Volume'].iloc[-1]) else 0
                        })
                
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
            
            # Sort by change percentage
            movers_data.sort(key=lambda x: x['change_pct'], reverse=True)
            
            # FIXED: Ensure gainers and losers are different
            gainers = [m for m in movers_data if m['change_pct'] > 0][:limit]
            losers = [m for m in movers_data if m['change_pct'] < 0][-limit:]
            losers.reverse()  # Show worst performers first
            
            return {
                'top_gainers': gainers,
                'top_losers': losers,
                'total_analyzed': len(movers_data)
            }
            
        except Exception as e:
            print(f"Top movers analysis error: {e}")
            return {'top_gainers': [], 'top_losers': [], 'total_analyzed': 0}
    
    def get_volatility_analysis(self, symbols: List[str]) -> Dict:
        """Analyze volatility for given stocks - FIXED"""
        try:
            volatility_data = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    hist = ticker.history(period='1mo')
                    
                    if len(hist) >= 20:
                        # Calculate daily returns
                        daily_returns = hist['Close'].pct_change().dropna()
                        
                        # Volatility metrics
                        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
                        avg_volume = hist['Volume'].mean() if 'Volume' in hist.columns else 0
                        
                        volatility_data.append({
                            'symbol': symbol,
                            'volatility': float(volatility * 100),  # As percentage
                            'avg_volume': float(avg_volume),
                            'price_range': float((hist['High'].max() - hist['Low'].min()) / hist['Close'].mean() * 100)
                        })
                
                except Exception:
                    continue
            
            # Sort by volatility
            volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
            
            # FIXED: Ensure high and low volatility are truly different
            if len(volatility_data) >= 2:
                median_volatility = np.median([v['volatility'] for v in volatility_data])
                
                high_vol = [v for v in volatility_data if v['volatility'] > median_volatility][:5]
                low_vol = [v for v in volatility_data if v['volatility'] <= median_volatility][-5:]
                low_vol.reverse()  # Show lowest volatility first
            else:
                high_vol = volatility_data[:5]
                low_vol = []
            
            return {
                'high_volatility': high_vol,
                'low_volatility': low_vol,
                'avg_market_volatility': float(np.mean([v['volatility'] for v in volatility_data])) if volatility_data else 0
            }
            
        except Exception as e:
            print(f"Volatility analysis error: {e}")
            return {'high_volatility': [], 'low_volatility': [], 'avg_market_volatility': 0}
    
    def get_correlation_analysis(self, symbols: List[str]) -> pd.DataFrame:
        """Get correlation matrix for given stocks - FIXED"""
        try:
            if len(symbols) < 2:
                return pd.DataFrame()
            
            # Fetch price data for all symbols
            price_data = {}
            
            for symbol in symbols[:10]:  # Limit to 10 for performance
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    hist = ticker.history(period='3mo')
                    if len(hist) >= 50:
                        price_data[symbol] = hist['Close']
                except Exception as e:
                    print(f"Error fetching correlation data for {symbol}: {e}")
                    continue
            
            if len(price_data) < 2:
                return pd.DataFrame()
            
            # Create DataFrame and calculate correlation
            df = pd.DataFrame(price_data)
            
            # Handle missing data
            df = df.dropna()
            
            if df.empty:
                return pd.DataFrame()
                
            correlation_matrix = df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            print(f"Correlation analysis error: {e}")
            return pd.DataFrame()

# Global instance
market_analyzer = MarketAnalyzer()
