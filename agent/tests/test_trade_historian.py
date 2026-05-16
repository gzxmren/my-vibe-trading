import pytest
import pandas as pd
from unittest.mock import patch
from scripts.gemini_engines.trade_historian import analyze_single_trade

@patch('scripts.gemini_engines.trade_historian.yf.Ticker')
def test_analyze_single_trade_basic(mock_ticker_class):
    # Mock the ticker and its history method
    mock_ticker = mock_ticker_class.return_value
    
    # Create a mock dataframe with enough data to calculate RSI and MA20
    dates = pd.date_range(start='2023-01-01', periods=30)
    mock_df = pd.DataFrame({
        'Close': [100.0 + i for i in range(30)]
    }, index=dates)
    
    mock_ticker.history.return_value = mock_df

    # 测试能否回溯 AMD 在特定时间点的数据
    result = analyze_single_trade(symbol="AMD", date_str="2023-05-15", action="BUY", price=420.0)
    assert "symbol" in result
    assert result["symbol"] == "AMD"
    assert "rsi_14" in result
    assert "price_to_ma20_bias" in result
