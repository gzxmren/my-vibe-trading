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

class StrategyCalculator:
    @staticmethod
    def calculate(strategy: Strategy) -> dict:
        net_premium = 0.0
        for leg in strategy.legs:
            if leg.action == "buy":
                net_premium -= leg.premium
            else:
                net_premium += leg.premium
                
        # Simplified max profit/loss calculation assuming vertical spreads for now.
        # Straddles/Iron Condors will need expanded logic here later, but this gets the spread working.
        net_cost = abs(net_premium) if net_premium < 0 else 0
        net_credit = net_premium if net_premium > 0 else 0
        
        # Calculate width of the spread
        strikes = sorted([leg.strike for leg in strategy.legs])
        width = strikes[-1] - strikes[0] if len(strikes) > 1 else 0
        
        if net_premium < 0: # Debit spread
            max_loss = net_cost
            max_profit = width - net_cost
        else: # Credit spread
            max_profit = net_credit
            max_loss = width - net_credit

        return {
            "net_cost": round(net_cost, 2),
            "net_credit": round(net_credit, 2),
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2)
        }
