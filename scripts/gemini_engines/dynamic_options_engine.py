from dataclasses import dataclass
from typing import List

@dataclass
class OptionLeg:
    strike: float
    option_type: str  # 'call' or 'put'
    action: str       # 'buy' or 'sell'
    premium: float
    implied_volatility: float = 0.0
    open_interest: int = 0
    
@dataclass
class Strategy:
    name: str
    legs: List[OptionLeg]
    
class RiskGuard:
    @staticmethod
    def is_defined_risk(strategy: Strategy) -> bool:
        # A simple heuristic: count short calls/puts vs long calls/puts.
        short_calls = sum(1 for leg in strategy.legs if leg.action == "sell" and leg.option_type == "call")
        long_calls = sum(1 for leg in strategy.legs if leg.action == "buy" and leg.option_type == "call")
        short_puts = sum(1 for leg in strategy.legs if leg.action == "sell" and leg.option_type == "put")
        long_puts = sum(1 for leg in strategy.legs if leg.action == "buy" and leg.option_type == "put")
        
        # Every short must be covered by a corresponding long
        if short_calls > long_calls: return False
        if short_puts > long_puts: return False
        return True
