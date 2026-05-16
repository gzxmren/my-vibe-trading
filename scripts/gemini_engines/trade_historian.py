#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import argparse
import json

def analyze_single_trade(symbol: str, date_str: str, action: str, price: float) -> dict:
    """回溯特定日期的股票技术指标"""
    try:
        # 获取包含历史数据（多取一些用于计算均线和RSI）
        ticker = yf.Ticker(symbol)
        df = ticker.history(end=date_str, period="3mo")
        if df.empty:
            return {"error": f"No data found for {symbol} up to {date_str}"}
        
        # 计算 RSI (简单版，真实开发可引入 ta 库)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算乖离率
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Bias20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        
        latest = df.iloc[-1]
        
        return {
            "symbol": symbol,
            "action": action,
            "exec_price": price,
            "close_price": latest['Close'],
            "rsi_14": latest.get('RSI', None),
            "price_to_ma20_bias": latest.get('Bias20', None)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade Historian (Sensory Component)")
    parser.add_argument("--symbol", type=str, help="Stock Symbol (e.g., AMD)")
    parser.add_argument("--date", type=str, help="Trade Date (YYYY-MM-DD)")
    parser.add_argument("--action", type=str, choices=["BUY", "SELL"])
    parser.add_argument("--price", type=float, help="Execution Price")
    
    args = parser.parse_args()
    if args.symbol and args.date and args.action and args.price:
        res = analyze_single_trade(args.symbol, args.date, args.action, args.price)
        print(json.dumps(res, indent=2))
