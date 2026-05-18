# Dynamic Options Strategy Generator Design Spec

## 1. Overview
A quantitative algorithmic engine integrated into `vibe-trading` to automatically evaluate and rank complex options strategies (Spreads, Straddles, Iron Condors). It bases recommendations on real-time Implied Volatility (IV) and Greeks, mitigating tail risks and completely eliminating the possibility of naked option selling.

## 2. Architecture & Components

### 2.1 Data Source Layer
*   **Provider**: `yfinance` library.
*   **Metrics Fetched**: Current asset price, options chain (`strike`, `bid`, `ask`, `lastPrice`, `impliedVolatility`, `openInterest`).
*   **Volatility Context**: Computes a basic IV Rank proxy (current IV relative to near-term historical IV, if available) to determine if options are currently "expensive" or "cheap".

### 2.2 Strategy Generation Engine
The engine programmatically constructs permutations of the following defined-risk strategies around the current At-The-Money (ATM) strike:
*   **Directional**: Bear Put Spread, Bear Call Spread, Bull Put Spread, Bull Call Spread.
*   **Volatility Expansion (Low IV environment)**: Long Straddle, Long Strangle.
*   **Volatility Contraction / Range-Bound (High IV environment)**: Iron Condor.

### 2.3 Intelligent Scoring & Ranking Algorithm
1.  **Liquidity Filter**: Discard any strategy where any leg has `openInterest` < 50 or the bid-ask spread is excessively wide (e.g., > 20% of the option price).
2.  **Risk/Reward Calculation**: For each valid permutation, calculate:
    *   Max Profit
    *   Max Loss (Net Debit or Margin Requirement)
    *   Break-even points
3.  **Scoring**: `Score = (Max Profit / Max Loss) * Volatility_Adjustment`.
    *   *Volatility_Adjustment* penalizes buying premium in high IV environments and penalizes selling premium in low IV environments.
4.  **Selection**: The engine sorts all valid permutations by Score and selects the Top 3 strategies for the user.

### 2.4 Risk Guard (Strict Constraint)
*   The system enforces a strict "Defined Risk Only" policy.
*   Any strategy resulting in infinite maximum loss (e.g., naked short call, naked short put) is structurally impossible to generate.

## 3. Data Flow
1. User invokes the script for a specific ticker (e.g., SMH).
2. Script fetches current price and available option expirations.
3. Selects the target expiration (~3 weeks out).
4. Generates all possible valid strike combinations for the supported strategies.
5. Calculates risk/reward metrics and scores for each combination.
6. Ranks and outputs a formatted, readable CLI report of the Top 3 strategies.

## 4. Error Handling
*   **API Failures**: Gracefully handle timeouts or missing data from `yfinance`, falling back to cached data if applicable or aborting cleanly.
*   **Missing Greeks**: If `impliedVolatility` is null from the API, default to a neutral Volatility_Adjustment factor (1.0) and warn the user.

## 5. Testing Strategy
*   **Math Validation**: Unit tests to verify exact Max Profit and Max Loss calculations for Iron Condors and Straddles using hardcoded mock prices.
*   **Risk Guard Validation**: Unit tests asserting that no generated strategy can output an infinite max loss.
