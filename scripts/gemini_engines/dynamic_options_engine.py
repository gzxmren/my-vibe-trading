import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Any

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
        if len(strategy.legs) != 2:
            raise NotImplementedError("Calculator currently only supports 2-leg vertical spreads.")

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

class OptionDataFetcher:
    @staticmethod
    def fetch_chain(ticker_symbol: str, target_days_out: int = 21):
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1d")
        if history.empty:
            return None, None, None
        current_price = history['Close'].iloc[-1]
        
        expirations = ticker.options
        if not expirations: return None, None, current_price
        
        target_date = datetime.now() + timedelta(days=target_days_out)
        target_exp = min(expirations, key=lambda d: abs(datetime.strptime(d, '%Y-%m-%d') - target_date))
        
        return ticker.option_chain(target_exp), target_exp, current_price

class StrategyGenerator:
    @staticmethod
    def generate_bear_put_spreads(chain: Any, current_price: float) -> List[Strategy]:
        puts = chain.puts
        # Basic filter: ATM and OTM puts
        atm_strike = round(current_price)
        valid_puts = puts[puts['strike'] <= atm_strike + 5]
        
        strategies = []
        # Create permutations of Bear Put Spreads
        # (Simplified iteration for the plan)
        for i, buy_leg_data in valid_puts.iterrows():
            for j, sell_leg_data in valid_puts.iterrows():
                if buy_leg_data['strike'] > sell_leg_data['strike']:
                    buy_leg = OptionLeg(buy_leg_data['strike'], "put", "buy", buy_leg_data['lastPrice'], buy_leg_data['impliedVolatility'], buy_leg_data['openInterest'])
                    sell_leg = OptionLeg(sell_leg_data['strike'], "put", "sell", sell_leg_data['lastPrice'], sell_leg_data['impliedVolatility'], sell_leg_data['openInterest'])
                    strat = Strategy(f"Bear Put Spread {buy_leg.strike}/{sell_leg.strike}", [buy_leg, sell_leg])
                    if RiskGuard.is_defined_risk(strat):
                        strategies.append(strat)
        return strategies

class ScoringEngine:
    @staticmethod
    def score_and_rank(strategies: List[Strategy]) -> List[dict]:
        scored = []
        for strat in strategies:
            # Liquidity check
            if any(leg.open_interest < 10 for leg in strat.legs):
                continue
                
            metrics = StrategyCalculator.calculate(strat)
            if metrics["max_loss"] <= 0: continue
            
            score = metrics["max_profit"] / metrics["max_loss"]
            
            scored.append({
                "strategy": strat,
                "metrics": metrics,
                "score": score
            })
            
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:3]

def main(ticker="SMH"):
    print(f"Fetching options data for {ticker}...")
    chain, exp_date, price = OptionDataFetcher.fetch_chain(ticker)
    if not chain: return
    
    print(f"Current Price: ${price:.2f} | Target Expiry: {exp_date}\n")
    
    spreads = StrategyGenerator.generate_bear_put_spreads(chain, price)
    top_3 = ScoringEngine.score_and_rank(spreads)
    
    print("=== TOP 3 STRATEGIES (Sorted by Risk/Reward Score) ===")
    for rank, item in enumerate(top_3, 1):
        strat = item["strategy"]
        metrics = item["metrics"]
        print(f"\n#{rank}: {strat.name}")
        for leg in strat.legs:
            print(f"  - {leg.action.upper()} {leg.strike} {leg.option_type.upper()} @ ${leg.premium:.2f} (IV: {leg.implied_volatility:.2f})")
        print(f"  -> Max Risk: ${metrics['max_loss']*100:.2f} | Max Reward: ${metrics['max_profit']*100:.2f}")
        print(f"  -> Score (RR): {item['score']:.2f}x")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SMH"
    main(ticker)

