import pytest
from scripts.gemini_engines.dynamic_options_engine import OptionLeg, Strategy, RiskGuard, StrategyCalculator

def test_risk_guard_rejects_infinite_loss():
    # A naked short call has infinite loss
    short_call = OptionLeg(strike=100, option_type="call", action="sell", premium=5.0)
    strategy = Strategy(name="Naked Short Call", legs=[short_call])
    
    assert RiskGuard.is_defined_risk(strategy) is False

def test_risk_guard_accepts_defined_risk():
    # A bear put spread has defined loss
    long_put = OptionLeg(strike=100, option_type="put", action="buy", premium=5.0)
    short_put = OptionLeg(strike=90, option_type="put", action="sell", premium=2.0)
    strategy = Strategy(name="Bear Put Spread", legs=[long_put, short_put])
    
    assert RiskGuard.is_defined_risk(strategy) is True

def test_strategy_calculator_bear_put():
    long_put = OptionLeg(strike=100, option_type="put", action="buy", premium=5.0)
    short_put = OptionLeg(strike=90, option_type="put", action="sell", premium=2.0)
    strategy = Strategy(name="Bear Put Spread", legs=[long_put, short_put])
    
    metrics = StrategyCalculator.calculate(strategy)
    assert metrics["net_cost"] == 3.0
    assert metrics["max_profit"] == 7.0 # width (10) - net_cost (3)
    assert metrics["max_loss"] == 3.0

