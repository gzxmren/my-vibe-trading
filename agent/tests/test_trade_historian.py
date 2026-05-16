import pytest
from scripts.gemini_engines.trade_historian import analyze_single_trade

def test_analyze_single_trade_basic():
    # 测试能否回溯 AMD 在特定时间点的数据
    result = analyze_single_trade(symbol="AMD", date_str="2026-05-15", action="BUY", price=420.0)
    assert "symbol" in result
    assert result["symbol"] == "AMD"
    assert "rsi_14" in result
    assert "price_to_ma20_bias" in result
