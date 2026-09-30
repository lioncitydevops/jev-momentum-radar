import os
import json
import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Global Multi-Index Momentum Radar - TypeSafe Jev",
    page_icon="🌐",
    layout="wide"
)

# ---------------------------------------------------------
# API Credentials
# ---------------------------------------------------------
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .index-card {
        padding: 18px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .index-card:hover {
        transform: translateY(-2px);
    }
    .card-buy {
        background: linear-gradient(135deg, #0e3a24 0%, #051d12 100%);
        border: 2px solid #00ff88;
        color: #ffffff;
    }
    .card-sell {
        background: linear-gradient(135deg, #3d1419 0%, #1f080b 100%);
        border: 2px solid #ff4d6d;
        color: #ffffff;
    }
    .card-neutral {
        background: linear-gradient(135deg, #382c0b 0%, #1c1503 100%);
        border: 2px solid #ffd166;
        color: #ffffff;
    }
    .metric-badge {
        font-size: 26px;
        font-weight: 800;
        margin: 6px 0;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        background-color: #0b2e1b;
        color: #00ff88;
        border: 1px solid #00ff88;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Asset Definitions
# ---------------------------------------------------------
ASSETS = {
    "S&P 500 (ES)": {"symbol": "ES=F", "flag": "🇺🇸", "desc": "E-mini S&P 500 Futures (CME)"},
    "Nasdaq 100 (NQ)": {"symbol": "NQ=F", "flag": "💻", "desc": "E-mini Nasdaq 100 Futures (CME)"},
    "Russell 2000 (RTY)": {"symbol": "RTY=F", "flag": "🚀", "desc": "E-mini Russell 2000 Futures (CME)"},
    "Nikkei 225 (NKD)": {"symbol": "NKD=F", "flag": "🇯🇵", "desc": "Nikkei 225 Index Futures (OSE/CME)"},
    "10Y T-Note (TY10)": {"symbol": "ZN=F", "flag": "🏛️", "desc": "10-Year Treasury Note Futures (CBOT)"}
}

# ---------------------------------------------------------
# Data Fetcher
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_asset_data(symbol: str, timeframe: str = "5m"):
    range_str = "5d" if timeframe == "5m" else ("1mo" if timeframe == "1h" else "3mo")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={timeframe}&range={range_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()["chart"]["result"][0]
    
    timestamps = data["timestamp"]
    quote = data["indicators"]["quote"][0]
    
    df = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote.get("volume", [0]*len(timestamps)),
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))
    
    df = df.dropna(subset=["Close"])
    return df

# ---------------------------------------------------------
# Live Jev Query Function
# ---------------------------------------------------------
def call_jev_api(state_str: str) -> dict:
    """Calls TypeSafe Jev System One API."""
    payload = {
        "model": "jev-latest",
        "state": state_str,
        "questions": {
            "tactical_action": {
                "type": "choice",
                "instructions": "Determine tactical positioning for the next 15-30 minutes based on momentum and VWAP.",
                "criteria": {
                    "BUY_LONG": "Clear bullish momentum above VWAP with upside acceleration",
                    "HOLD_CASH": "Consolidation, neutral chop, or tight range near VWAP",
                    "SELL_SHORT": "Clear bearish breakdown below VWAP with downside acceleration"
                }
            },
            "prob_continuation": {
                "type": "noul",
                "instructions": "Will price close higher over the next 3 bars?"
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {TYPESAFE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    resp = requests.post(TYPESAFE_URL, json=payload, headers=headers, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    
    answers = data.get("answers", {})
    action_data = answers.get("tactical_action", {})
    prob_data = answers.get("prob_continuation", {})
    
    choice = action_data.get("choice", "HOLD_CASH")
    confidence = action_data.get("confidence", 0.6)
    prob_noul = prob_data.get("noul", 0.50)
    
    return {
        "choice": choice,
        "confidence": confidence,
        "prob_up": prob_noul,
        "raw_response": data
    }

# ---------------------------------------------------------
# Feature & Signal Computation
# ---------------------------------------------------------
def analyze_asset(df, name="S&P 500", timeframe="5m"):
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    open_p = df["Open"].values
    volume = df["Volume"].fillna(0).values
    
    if len(close) < 15:
        return None

    total_vol = volume.sum()
    if total_vol > 0:
        typical_price = (high + low + close) / 3.0
        vwap = (typical_price * volume).sum() / (total_vol + 1e-9)
    else:
        vwap = float(np.mean(close[-20:]))
        
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.std(close))
    vwap_z = (close[-1] - vwap) / (atr_14 + 1e-9)
    
    ret_3 = (close[-1] / close[-4] - 1) * 100 if len(close) > 4 else 0.0
    ret_6 = (close[-1] / close[-7] - 1) * 100 if len(close) > 7 else 0.0
    session_change = (close[-1] / open_p[0] - 1) * 100
    
    # 14-period RSI
    delta = np.diff(close[-15:])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    rs = np.mean(gain) / (np.mean(loss) + 1e-9)
    rsi_14 = 100 - (100 / (1 + rs))
    
    state_str = (
        f"Asset: {name} ({timeframe} bar). "
        f"Close Price: ${close[-1]:,.2f}. "
        f"Session VWAP: ${vwap:,.2f} (Distance: {vwap_z:+.2f} ATRs). "
        f"Micro-Momentum: 3-bar={ret_3:+.2f}%, 6-bar={ret_6:+.2f}%. "
        f"14-RSI: {rsi_14:.1f}. Session Return: {session_change:+.2f}%."
    )
    
    # Execute Live Jev call
    try:
        jev_res = call_jev_api(state_str)
        jev_choice = jev_res["choice"]
        prob_up = jev_res["prob_up"]
        confidence = jev_res["confidence"]
        is_live_jev = True
    except Exception as e:
        # Fallback to calibrated quant heuristic if API limit or error
        raw_score = 0.35 * ret_3 + 0.35 * ret_6 + 0.30 * (vwap_z * 0.4)
        prob_up = float(1.0 / (1.0 + np.exp(- (0.19 + 0.35 * raw_score))))
        jev_choice = "BUY_LONG" if (prob_up > 0.54 and vwap_z > 0.15) else ("SELL_SHORT" if (prob_up < 0.46 and vwap_z < -0.15) else "HOLD_CASH")
        confidence = 0.70
        is_live_jev = False

    if "BUY" in jev_choice:
        action = "BUY / LONG"
        badge_class = "card-buy"
        color = "#00ff88"
    elif "SELL" in jev_choice:
        action = "SELL / SHORT"
        badge_class = "card-sell"
        color = "#ff4d6d"
    else:
        action = "HOLD CASH"
        badge_class = "card-neutral"
        color = "#ffd166"
        
    return {
        "action": action,
        "badge_class": badge_class,
        "color": color,
        "prob_up": prob_up,
        "confidence": confidence,
        "is_live_jev": is_live_jev,
        "price": close[-1],
        "session_change": session_change,
        "vwap": vwap,
        "vwap_z": vwap_z,
        "rsi": rsi_14,
        "ret_3": ret_3,
        "ret_6": ret_6,
        "state_str": state_str,
        "timestamp": df.index[-1].strftime("%Y-%m-%d %H:%M UTC")
    }

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🌐 Global Multi-Index Momentum Radar")
    st.caption("Nikkei 225 • S&P 500 • Nasdaq 100 • Russell 2000 — Powered by TypeSafe Jev System One")
with header_col2:
    st.write("")
    st.markdown('<div style="text-align: right;"><span class="status-badge">🟢 JEV MODEL: ONLINE (jev-1.13.0)</span></div>', unsafe_allow_html=True)

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

with ctrl_col1:
    timeframe = st.selectbox(
        "Timeframe Horizon",
        ["5m (5-Minute Intraday Scalp)", "1h (1-Hour Tactical Swing)", "1d (Daily Holding)"],
        index=0
    )
    tf_code = "5m" if "5m" in timeframe else ("1h" if "1h" in timeframe else "1d")

with ctrl_col2:
    st.write("")
    st.write("")
    st.button("🔄 Refresh Radar Now", use_container_width=True)

with ctrl_col3:
    auto_refresh = st.checkbox("Auto-refresh (every 60s)", value=False)

# Fetch and analyze all assets
results = {}
raw_dfs = {}
for name, meta in ASSETS.items():
    try:
        df = fetch_asset_data(meta["symbol"], tf_code)
        sig = analyze_asset(df, name=name, timeframe=tf_code)
        if sig:
            results[name] = sig
            raw_dfs[name] = df
    except Exception as e:
        st.warning(f"Error analyzing {name}: {e}")

# ---------------------------------------------------------
# 1. Multi-Index Cards Grid
# ---------------------------------------------------------
st.subheader("⚡ Live Decision Overview (TypeSafe Jev)")
cols = st.columns(4)

for i, (name, meta) in enumerate(ASSETS.items()):
    sig = results.get(name)
    with cols[i]:
        if sig:
            engine_tag = "Jev Decision" if sig["is_live_jev"] else "Quant Fallback"
            st.markdown(f"""
            <div class="index-card {sig['badge_class']}">
                <div style="font-size: 13px; opacity: 0.8;">{meta['flag']} {name} ({meta['symbol']})</div>
                <div style="font-size: 22px; font-weight: 700; margin: 4px 0;">${sig['price']:,.2f}</div>
                <div style="font-size: 12px; margin-bottom: 6px; color: {'#00ff88' if sig['session_change'] >= 0 else '#ff4d6d'};">
                    {sig['session_change']:+.2f}% (Session)
                </div>
                <div class="metric-badge" style="color: {sig['color']};">{sig['action']}</div>
                <div style="font-size: 13px; font-weight: 600;">Continuation Prob: {sig['prob_up']*100:.1f}%</div>
                <div style="font-size: 11px; opacity: 0.75; margin-top: 4px;">Confidence: {sig['confidence']*100:.0f}% | VWAP: {sig['vwap_z']:+.1f}σ</div>
                <div style="font-size: 10px; opacity: 0.6; margin-top: 4px;">⚡ {engine_tag}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Loading {name}...")

st.divider()

# ---------------------------------------------------------
# 2. Cross-Market Relative Strength Matrix
# ---------------------------------------------------------
st.subheader("📊 Cross-Market Relative Strength Matrix")
matrix_data = []
for name, sig in results.items():
    matrix_data.append({
        "Index": f"{ASSETS[name]['flag']} {name}",
        "Signal": sig["action"],
        "Continuation Prob": f"{sig['prob_up']*100:.1f}%",
        "Jev Confidence": f"{sig['confidence']*100:.0f}%",
        "3-Bar Momentum": f"{sig['ret_3']:+.2f}%",
        "6-Bar Momentum": f"{sig['ret_6']:+.2f}%",
        "VWAP Distance": f"{sig['vwap_z']:+.2f} ATR",
        "14-RSI": f"{sig['rsi']:.1f}"
    })

matrix_df = pd.DataFrame(matrix_data)
st.dataframe(matrix_df, use_container_width=True, hide_index=True)

# Cross-Market Intelligence
if len(results) == 4:
    sp = results["S&P 500"]
    ndx = results["Nasdaq 100"]
    rut = results["Russell 2000"]
    nik = results["Nikkei 225"]
    
    tech_spread = ndx["ret_6"] - sp["ret_6"]
    small_spread = rut["ret_6"] - sp["ret_6"]
    
    insights = []
    if tech_spread > 0.15:
        insights.append("🟢 **Tech Outperformance (QQQ > SPY)**: Tech leadership is driving index momentum.")
    elif tech_spread < -0.15:
        insights.append("🟡 **Tech Drag (QQQ < SPY)**: Tech sector lagging broader market.")
        
    if small_spread > 0.20:
        insights.append("🚀 **Broad Risk-On (IWM > SPY)**: Small caps showing high-beta participation.")
    elif small_spread < -0.20:
        insights.append("⚠️ **Defensive Posture (IWM < SPY)**: Small caps underperforming; watch for false large-cap breakouts.")
        
    if nik["prob_up"] > 0.55:
        insights.append("🇯🇵 **Nikkei Momentum**: Asian session trading with upside momentum bias.")
        
    if insights:
        st.info("💡 **Cross-Asset Intelligence:**\n\n" + "\n\n".join(insights))

st.divider()

# ---------------------------------------------------------
# 3. Interactive Candlestick Chart
# ---------------------------------------------------------
selected_asset = st.selectbox("Select Asset for Deep-Dive Chart", list(ASSETS.keys()), index=0)
if selected_asset in raw_dfs:
    chart_df = raw_dfs[selected_asset].iloc[-45:]
    sel_sig = results[selected_asset]
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'],
        high=chart_df['High'],
        low=chart_df['Low'],
        close=chart_df['Close'],
        name=selected_asset
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=[sel_sig['vwap']] * len(chart_df),
        mode='lines',
        line=dict(color='yellow', width=2, dash='dash'),
        name="Session VWAP"
    ))
    fig.update_layout(
        title=f"{ASSETS[selected_asset]['flag']} {selected_asset} ({ASSETS[selected_asset]['symbol']}) - {timeframe}",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()
