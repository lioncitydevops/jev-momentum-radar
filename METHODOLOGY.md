# 📐 Quantitative Methodology Note: Short-Term Multi-Futures Momentum Radar

**Engine**: TypeSafe Jev System One (`jev-latest`)  
**Data Infrastructure**: TradingView Official Real-Time Futures Scanner & Streaming Feeds  
**Asset Universe**: 24-Hour Continuous Benchmark Futures (`ES1!`, `NQ1!`, `RTY1!`, `NK2251!`, `ZN1!`)  
**Deployment**: Serverless FastAPI Python 3.12 Engine on Vercel  

---

## 1. Executive Overview & System Architecture

The **Global Multi-Futures Momentum Radar** is a dual-layer quantitative decision architecture. It combines **continuous mathematical feature extraction** across multi-horizon market microstructure with **TypeSafe Jev System One** models to produce probabilistic tactical judgments (`BUY / LONG`, `SELL / SHORT`, `HOLD CASH`) with Bayesian-calibrated continuation probabilities.

```mermaid
flowchart TD
    subgraph MarketFeeds ["1. Real-Time Market Data Ingestion"]
        ES["E-mini S&P 500 (CME: ES1!)"]
        NQ["E-mini Nasdaq 100 (CME: NQ1!)"]
        RTY["E-mini Russell 2000 (CME: RTY1!)"]
        NKD["Nikkei 225 Futures (OSE: NK2251!)"]
        ZN["10Y Treasury Note (CBOT: ZN1!)"]
    end

    subgraph FeatureEngineering ["2. Mathematical Feature Engineering"]
        VWAP["Anchored Session VWAP & ATR-z Spread"]
        MOM["Multi-Scale Momentum Z-Scores (3b, 6b, 12b)"]
        RSI["14-Period Smoothed Wilder RSI"]
        TVRating["TradingView Multi-Indicator Rating (-1.0 to +1.0)"]
        RVOL["Relative Volume (RVOL) Flow"]
    end

    subgraph DecisionLayer ["3. TypeSafe Jev System One Decision Engine"]
        StatePrompt["Structured Quantitative State Prompt"]
        JevAPI["TypeSafe Jev System One (jev-latest)"]
        Choice["Tactical Action Choice (BUY / SELL / HOLD)"]
        Noul["Probabilistic Continuation Metric (0.0 to 1.0)"]
    end

    subgraph ExecutionLayer ["4. Web Radar & Alert Delivery"]
        VercelAPI["Serverless FastAPI on Vercel"]
        Dashboard["Interactive Glassmorphism Dashboard"]
        TVWebhook["Pine Script Alert Webhook Bridge"]
    end

    MarketFeeds --> FeatureEngineering
    FeatureEngineering --> StatePrompt
    StatePrompt --> JevAPI
    JevAPI --> Choice & Noul
    Choice & Noul --> VercelAPI
    VercelAPI --> Dashboard & TVWebhook
```

---

## 2. Mathematical Feature Engineering

Raw tick and candle data are converted into scale-invariant, normalized state representations to eliminate price-level bias and heteroskedasticity.

### 2.1 Anchored Session VWAP & Volatility Distance ($Z_{VWAP}$)
Intraday fair value is anchored to the opening tick of the continuous session using Volume-Weighted Average Price:

$$\text{VWAP}_t = \frac{\sum_{i=1}^{t} P_{\text{typical}, i} \cdot V_i}{\sum_{i=1}^{t} V_i}$$

where $P_{\text{typical}} = \frac{\text{High} + \text{Low} + \text{Close}}{3}$.

To measure market stretch independent of asset denomination, the distance between current close $P_t$ and $\text{VWAP}_t$ is normalized by the 14-period Average True Range ($\text{ATR}_{14}$):

$$Z_{\text{VWAP}} = \frac{P_t - \text{VWAP}_t}{\text{ATR}_{14} + \epsilon}$$

* **$Z_{\text{VWAP}} > +1.5\sigma$**: Extended upside momentum (continuation risk increases; pullback risk emerges).
* **$-0.5\sigma \le Z_{\text{VWAP}} \le +0.5\sigma$**: Equilibrium zone (trend-following re-test entries).
* **$Z_{\text{VWAP}} < -1.5\sigma$**: Breakdown impulse (bearish momentum acceleration).

---

### 2.2 Volatility-Adjusted Multi-Scale Momentum ($Z_{\text{mom}}$)
Short-term momentum is calculated across three lookback horizons corresponding to micro-tactical timeframes:
* **3-bar micro-burst**: $15\text{m}$ on a $5\text{m}$ chart
* **6-bar session momentum**: $30\text{m}$ on a $5\text{m}$ chart
* **12-bar cycle trend**: $60\text{m}$ on a $5\text{m}$ chart

Logarithmic returns are normalized using Garman-Klass realized daily volatility ($\sigma_{GK}$):

$$\sigma_{GK}^2 = \frac{1}{N} \sum_{i=1}^{N} \left[ 0.5 \left(\ln \frac{H_i}{L_i}\right)^2 - (2\ln 2 - 1) \left(\ln \frac{C_i}{O_i}\right)^2 \right]$$

$$Z_{\text{mom}, h} = \frac{\ln(C_t / C_{t-h})}{\sigma_{GK} \sqrt{h}}$$

---

### 2.3 TradingView Multi-Indicator Technical Rating ($\text{Rating}_{TV}$)
The system extracts TradingView’s real-time quantitative consensus score $\in [-1.0, +1.0]$, aggregating:
* **16 Moving Average Filters**: Exponential (EMA 10, 20, 30, 50, 100, 200), Simple (SMA), Hull MA, and Ichimoku Baseline.
* **11 Momentum Oscillators**: 14-period RSI, Stochastic %K/%D, MACD Histogram, Commodity Channel Index (CCI20), Average Directional Index (ADX), and Awesome Oscillator.

$$\text{Rating}_{TV} = w_{MA} \cdot \text{Score}_{MA} + w_{Osc} \cdot \text{Score}_{Osc}$$

| Score Range | Technical Classification | Color Mapping |
| :--- | :--- | :--- |
| $+0.50 \text{ to } +1.00$ | **STRONG BUY** | Emerald Neon (`#00ff88`) |
| $+0.10 \text{ to } +0.49$ | **BUY** | Pine Green (`#22c55e`) |
| $-0.09 \text{ to } +0.09$ | **NEUTRAL** | Amber Gold (`#ffd166`) |
| $-0.49 \text{ to } -0.10$ | **SELL** | Bright Red (`#ef4444`) |
| $-1.00 \text{ to } -0.50$ | **STRONG SELL** | Crimson Neon (`#ff4d6d`) |

---

### 2.4 Relative Volume (RVOL)
Measures institutional flow intensity compared to a 20-period moving average:

$$\text{RVOL}_t = \frac{V_t}{\frac{1}{20} \sum_{i=0}^{19} V_{t-i}}$$

$\text{RVOL} > 1.5\times$ indicates high-conviction institutional participation during breakouts or breakdowns.

---

## 3. TypeSafe Jev System One Decision Engine

Instead of rigid linear rules or brittle black-box weights, the synthesized feature vector is converted into a **structured state representation** passed to **TypeSafe Jev System One** (`https://api.typesafe.ai/v1/systemone`).

### 3.1 State Representation Prompt
```text
Futures Asset: S&P 500 (ES) (TradingView Symbol: CME_MINI:ES1!, 5m horizon).
TradingView Futures Price: $7,749.00.
TradingView Futures VWAP: $7,746.25 (Distance: +0.73 ATRs).
TradingView 14-RSI: 53.2.
TradingView Technical Rating: STRONG BUY (+0.54).
Micro-Momentum: 3-bar=+0.22%, 6-bar=+0.18%. Session Return: +0.22%.
```

### 3.2 Decision Questions & Primitives
Jev evaluates two formal System One primitives:

1. **Primitive: `choice` (`tactical_action`)**
   * *Instructions*: "Determine tactical positioning for the next 15-30 minutes based on TradingView futures momentum, VWAP deviation, and technical ratings."
   * *Options*:
     * `BUY_LONG`: Bullish momentum acceleration above VWAP with upside volume flow.
     * `SELL_SHORT`: Bearish breakdown acceleration below VWAP with liquidation flow.
     * `HOLD_CASH`: Mean-reverting consolidation, equilibrium chop, or ambiguous momentum.

2. **Primitive: `noul` (`prob_continuation`)**
   * *Instructions*: "Will price close higher over the next 3 bars?"
   * *Output*: Continuous probability $P(\text{Up}) \in [0.0, 1.0]$.

---

## 4. Cross-Market & Macro Yield Spread Engine

The radar integrates cross-market asset spreads across equities and sovereign debt:

$$\text{Tech Spread} = \text{ret}_{6}(\text{NQ1!}) - \text{ret}_{6}(\text{ES1!})$$

$$\text{Beta Spread} = \text{ret}_{6}(\text{RTY1!}) - \text{ret}_{6}(\text{ES1!})$$

$$\text{Bond-Equity Correlation} = \text{Sign}\left(\text{ret}_{6}(\text{ZN1!}) \cdot \text{ret}_{6}(\text{ES1!})\right)$$

### Tactical Macro Regimes Identified:
1. **Flight-to-Safety Regime**:
   * Condition: $\text{ZN1! (10Y T-Note)} \uparrow$ while $\text{ES1! (S&P 500)} \downarrow$
   * Interpretation: Capital rotating out of risk assets into safe-haven duration.
2. **Growth Reflation Regime**:
   * Condition: $\text{ES1!} \uparrow$, $\text{NQ1!} \uparrow$, and $\text{ZN1!} \downarrow$
   * Interpretation: Risk-on economic expansion; equity multiple tolerance high despite firming yields.
3. **Rate Pressure / Duration Drag**:
   * Condition: $\text{ZN1! Technical Rating} \le -0.25$ (Yields surging)
   * Interpretation: Duration risk; headwind for high-multiple Nasdaq futures (`NQ1!`).

---

## 5. Walk-Forward Calibration & Fallback Layer

To prevent operational downtime if external LLM gateways experience latency or rate limits, the system features a **purged and embargoed logistic calibration model** based on Marcos López de Prado's *Advances in Financial Machine Learning*:

$$P(\text{Up}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot \text{RawScore})}}$$

$$\text{RawScore} = 0.35 \cdot Z_{\text{mom}, 3} + 0.35 \cdot Z_{\text{mom}, 6} + 0.30 \cdot (0.4 \cdot Z_{\text{VWAP}}) + 0.20 \cdot \text{Rating}_{TV}$$

* **Walk-Forward Validation**: 5-year rolling training window with a 3-day embargo window to eliminate lookahead leakage.
* **Accuracy Thresholds**: Action triggers require directional probability $P(\text{Up}) > 0.54$ with $Z_{\text{VWAP}} > +0.15\sigma$ for Longs, and $P(\text{Up}) < 0.46$ with $Z_{\text{VWAP}} < -0.15\sigma$ for Shorts.

---

## 6. Real-Time Data Pipeline Summary

```
TradingView Futures Scanner (https://scanner.tradingview.com/futures/scan)
  │
  ├──> Real-Time Prices, VWAP, 14-RSI, Rating, Volume
  │
FastAPI Backend (api/index.py on Vercel)
  │
  ├──> Computes ATR-z, Momentum Spreads, Macro Inter-market Signals
  │
TypeSafe Jev System One (https://api.typesafe.ai/v1/systemone)
  │
  ├──> Generates Tactical Choice (BUY / SELL / HOLD) & Continuation Prob %
  │
Web Radar Dashboard (https://jevshorttermmomentum.vercel.app)
  │
  ├──> Real-Time Glassmorphic Cards, Relative Strength Matrix,
  └──> Official Embedded TradingView Advanced Streaming Chart (tv.js)
```
