import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple
import streamlit as st

class MarketAnalyzer:
    """Advanced market analysis with proper data separation - FIXED VERSION"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def get_sector_performance(self, symbols: List[str]) -> Dict:
        """Analyze sector performance for given symbols"""
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
        """Get top gainers and losers - FIXED to ensure distinct lists"""
        try:
            gainers_list = []
            losers_list = []
            all_data = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    hist = ticker.history(period='2d')
                    
                    if len(hist) >= 2:
                        current_price = float(hist['Close'].iloc[-1])
                        prev_close = float(hist['Close'].iloc[-2])
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        stock_data = {
                            'symbol': symbol,
                            'price': current_price,
                            'change_pct': change_pct,
                            'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns and not pd.isna(hist['Volume'].iloc[-1]) else 0
                        }
                        
                        all_data.append(stock_data)
                        
                        # FIXED: Separate into distinct lists based on performance
                        if change_pct > 0:
                            gainers_list.append(stock_data)
                        elif change_pct < 0:
                            losers_list.append(stock_data)
                
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
            
            # FIXED: Sort and limit each list separately
            top_gainers = sorted(gainers_list, key=lambda x: x['change_pct'], reverse=True)[:limit]
            top_losers = sorted(losers_list, key=lambda x: x['change_pct'], reverse=False)[:limit]  # Worst performers first
            
            print(f"DEBUG: Found {len(gainers_list)} gainers, {len(losers_list)} losers")
            print(f"DEBUG: Returning {len(top_gainers)} top gainers, {len(top_losers)} top losers")
            
            return {
                'top_gainers': top_gainers,
                'top_losers': top_losers,
                'total_analyzed': len(all_data)
            }
            
        except Exception as e:
            print(f"Top movers analysis error: {e}")
            return {'top_gainers': [], 'top_losers': [], 'total_analyzed': 0}
    
    def get_volatility_analysis(self, symbols: List[str]) -> Dict:
        """Analyze volatility - FIXED to ensure distinct high/low lists"""
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
            
            if not volatility_data:
                return {'high_volatility': [], 'low_volatility': [], 'avg_market_volatility': 0}
            
            # Sort by volatility
            volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
            
            # FIXED: Use threshold-based separation to ensure distinct lists
            volatility_values = [v['volatility'] for v in volatility_data]
            median_volatility = np.median(volatility_values)
            
            # Split based on median threshold
            high_volatility = [v for v in volatility_data if v['volatility'] > median_volatility]
            low_volatility = [v for v in volatility_data if v['volatility'] <= median_volatility]
            
            # Take top 5 from each category
            high_vol_top = high_volatility[:5]
            low_vol_top = sorted(low_volatility, key=lambda x: x['volatility'])[:5]
            
            print(f"DEBUG: Median volatility: {median_volatility:.2f}%")
            print(f"DEBUG: High vol stocks: {len(high_volatility)}, Low vol stocks: {len(low_volatility)}")
            
            return {
                'high_volatility': high_vol_top,
                'low_volatility': low_vol_top,
                'avg_market_volatility': float(np.mean(volatility_values))
            }
            
        except Exception as e:
            print(f"Volatility analysis error: {e}")
            return {'high_volatility': [], 'low_volatility': [], 'avg_market_volatility': 0}
    
    def get_correlation_analysis(self, symbols: List[str]) -> pd.DataFrame:
        """Get correlation matrix - FIXED to return proper correlation data"""
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
    
    def get_correlation_insights(self, correlation_matrix: pd.DataFrame) -> Tuple[List, List]:
        """Extract positive and negative correlation insights - NEW METHOD"""
        try:
            if correlation_matrix.empty:
                return [], []
            
            positive_correlations = []
            negative_correlations = []
            
            # Extract correlation pairs
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    stock1 = correlation_matrix.columns[i]
                    stock2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    
                    # Skip NaN correlations
                    if pd.isna(corr_value):
                        continue
                    
                    correlation_pair = (stock1, stock2, float(corr_value))
                    
                    # FIXED: Separate positive and negative correlations
                    if corr_value >= 0.1:  # Minimum threshold for positive
                        positive_correlations.append(correlation_pair)
                    elif corr_value <= -0.1:  # Maximum threshold for negative
                        negative_correlations.append(correlation_pair)
            
            # Sort correlations
            positive_correlations.sort(key=lambda x: x[2], reverse=True)  # Highest positive first
            negative_correlations.sort(key=lambda x: x[2])  # Lowest (most negative) first
            
            # Return top 5 of each
            return positive_correlations[:5], negative_correlations[:5]
            
        except Exception as e:
            print(f"Correlation insights error: {e}")
            return [], []

# Global instance
market_analyzer = MarketAnalyzer()
