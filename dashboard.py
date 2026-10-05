import os
import json
import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from oanda_feed import (
    fetch_oanda_candles,
    test_oanda_connection,
    get_configured_credentials,
    OANDA_INSTRUMENTS,
    SYMBOL_TO_OANDA
)

load_dotenv()

st.set_page_config(
    page_title="Global Multi-CFD Momentum Radar - TypeSafe Jev",
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
    .feed-badge-live {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 700;
        background-color: rgba(0, 255, 136, 0.15);
        color: #00ff88;
        border: 1px solid #00ff88;
        margin-top: 4px;
    }
    .feed-badge-delayed {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 700;
        background-color: rgba(255, 209, 102, 0.15);
        color: #ffd166;
        border: 1px solid #ffd166;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Asset Definitions (100% CFD Tickers with OANDA Support)
# ---------------------------------------------------------
ASSETS = {
    "S&P 500 (SPX)": {
        "symbol": "^GSPC",
        "oanda": "SPX500_USD",
        "flag": "🇺🇸",
        "desc": "S&P 500 Index CFD"
    },
    "Nasdaq 100 (NDX)": {
        "symbol": "^NDX",
        "oanda": "NAS100_USD",
        "flag": "💻",
        "desc": "Nasdaq 100 Index CFD"
    },
    "Russell 2000 (RUT)": {
        "symbol": "^RUT",
        "oanda": "US2000_USD",
        "flag": "🚀",
        "desc": "Russell 2000 Index CFD"
    },
    "Nikkei 225 (NI225)": {
        "symbol": "^N225",
        "oanda": "JP225_USD",
        "flag": "🇯🇵",
        "desc": "Nikkei 225 Index CFD"
    },
    "10Y Treasury (TNX)": {
        "symbol": "^TNX",
        "oanda": "USB10Y_USD",
        "flag": "🏛️",
        "desc": "10-Year U.S. Treasury Note CFD"
    },
    "WTI Crude (WTI)": {
        "symbol": "CL=F",
        "oanda": "WTICO_USD",
        "flag": "🛢️",
        "desc": "WTI Light Sweet Crude Oil CFD"
    },
    "Brent Crude (BRENT)": {
        "symbol": "BZ=F",
        "oanda": "BCO_USD",
        "flag": "🌊",
        "desc": "Brent North Sea Crude Oil CFD"
    }
}

# ---------------------------------------------------------
# Sidebar Configuration: Data Feed Selection
# ---------------------------------------------------------
st.sidebar.markdown("### 📡 Market Data Feed")

env_oanda_key, env_oanda_type = get_configured_credentials()

feed_choice = st.sidebar.radio(
    "Active Price Provider:",
    ["🟢 OANDA CFD (Zero-Lag, 24/5)", "⚠️ Yahoo Finance (15m Delayed)"],
    index=0 if env_oanda_key else 0
)

is_oanda_selected = "OANDA" in feed_choice

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ OANDA Settings")

oanda_key_input = st.sidebar.text_input(
    "OANDA v20 API Token",
    value=env_oanda_key,
    type="password",
    help="Personal Access Token from your OANDA Practice or Live account."
)

oanda_env_input = st.sidebar.selectbox(
    "OANDA Environment",
    ["practice (Demo Account)", "live (Real Account)"],
    index=0 if env_oanda_type == "practice" else 1
)
chosen_env = "practice" if "practice" in oanda_env_input else "live"

col_save, col_test = st.sidebar.columns([1, 1])

if col_save.button("💾 Save Key", use_container_width=True):
    try:
        # Read current .env
        env_lines = []
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                env_lines = f.readlines()
        
        new_lines = []
        has_key = False
        has_env = False
        for line in env_lines:
            if line.startswith("OANDA_API_KEY="):
                new_lines.append(f"OANDA_API_KEY={oanda_key_input.strip()}\n")
                has_key = True
            elif line.startswith("OANDA_ENVIRONMENT="):
                new_lines.append(f"OANDA_ENVIRONMENT={chosen_env}\n")
                has_env = True
            else:
                new_lines.append(line)
                
        if not has_key:
            new_lines.append(f"OANDA_API_KEY={oanda_key_input.strip()}\n")
        if not has_env:
            new_lines.append(f"OANDA_ENVIRONMENT={chosen_env}\n")
            
        with open(".env", "w") as f:
            f.writelines(new_lines)
            
        os.environ["OANDA_API_KEY"] = oanda_key_input.strip()
        os.environ["OANDA_ENVIRONMENT"] = chosen_env
        st.sidebar.success("Saved to .env!")
    except Exception as e:
        st.sidebar.error(f"Save failed: {e}")

if col_test.button("🧪 Test OANDA", use_container_width=True):
    ok, msg = test_oanda_connection(oanda_key_input.strip(), chosen_env)
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

with st.sidebar.expander("ℹ️ How to get a FREE OANDA API Key"):
    st.markdown("""
    1. Visit [oanda.com](https://www.oanda.com) and create a **Free Practice / Demo Account** (takes 60 seconds, no deposit or ID required).
    2. Log into the OANDA Hub / Portal.
    3. Navigate to **Manage API Access** -> **Generate Token**.
    4. Copy the Personal Access Token and paste it above.
    5. Enjoy instant sub-second, continuous 24/5 CFD feeds for S&P 500, Nasdaq, Nikkei, and Treasuries!
    """)

# ---------------------------------------------------------
# Data Fetchers
# ---------------------------------------------------------
@st.cache_data(ttl=5)
def fetch_oanda_data(instrument: str, timeframe: str = "1m", api_key: str = "", environment: str = "live"):
    count = 120 if timeframe == "1m" else (100 if timeframe == "5m" else (150 if timeframe == "1h" else 60))
    return fetch_oanda_candles(
        instrument=instrument,
        timeframe=timeframe,
        count=count,
        api_key=api_key,
        environment=environment
    )

@st.cache_data(ttl=30)
def fetch_yahoo_data(symbol: str, timeframe: str = "1m"):
    range_str = "1d" if timeframe == "1m" else ("5d" if timeframe == "5m" else ("1mo" if timeframe == "1h" else "3mo"))
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

def fetch_asset_data_unified(name: str, meta: dict, timeframe: str = "1m"):
    """
    Fetches price data according to selected feed, with intelligent fallback.
    """
    effective_key = oanda_key_input.strip() or os.getenv("OANDA_API_KEY", "")
    
    if is_oanda_selected and effective_key:
        try:
            df = fetch_oanda_data(
                instrument=meta["oanda"],
                timeframe=timeframe,
                api_key=effective_key,
                environment=chosen_env
            )
            return df, "OANDA (Zero-Lag Real-Time)"
        except Exception as e:
            # Fallback to Yahoo if OANDA fails
            df = fetch_yahoo_data(meta["symbol"], timeframe)
            return df, f"Yahoo Fallback (OANDA Error: {e})"
    else:
        df = fetch_yahoo_data(meta["symbol"], timeframe)
        return df, "Yahoo Finance (15m Delay)"

# ---------------------------------------------------------
# Live Jev Query Function (10-Minute Forward Horizon)
# ---------------------------------------------------------
def call_jev_api(state_str: str) -> dict:
    """Calls TypeSafe Jev System One API to forecast next 10-minute price trajectory."""
    payload = {
        "model": "jev-latest",
        "state": state_str,
        "questions": {
            "tactical_action": {
                "type": "choice",
                "instructions": "Determine tactical positioning for the next 10 minutes (next 10 bars forward on 1-minute interval) based on microstructure momentum, VWAP deviation, and short-term trend for CFD instruments.",
                "criteria": {
                    "BUY_LONG": "Clear bullish momentum above VWAP with high probability of higher prices over the next 10 minutes",
                    "HOLD_CASH": "Choppy, consolidation, neutral range, or tight oscillation near VWAP expected over next 10 minutes",
                    "SELL_SHORT": "Clear bearish breakdown below VWAP with high probability of lower prices over the next 10 minutes"
                }
            },
            "prob_continuation": {
                "type": "noul",
                "instructions": "What is the probability that price will close higher 10 minutes from now (next 10 bars forward on 1-minute interval)?"
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

from multi_horizon_momentum import generate_multi_horizon_signals
from macro_conditioned_momentum import generate_macro_full_signals, generate_rate_only_signals

def format_horizon_badge(action_str: str):
    if "LONG" in action_str or "BUY" in action_str:
        return action_str.replace("_", " "), "card-buy", "#00ff88"
    elif "SHORT" in action_str or "SELL" in action_str:
        return action_str.replace("_", " "), "card-sell", "#ff4d6d"
    else:
        return "NEUTRAL", "card-neutral", "#ffd166"

# ---------------------------------------------------------
# Feature & Signal Computation
# ---------------------------------------------------------
def analyze_asset(df, name="S&P 500 (SPX)", timeframe="1m", feed_source="OANDA", model_variant="standalone", macro_dfs=None):
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
    
    ret_3 = ((close[-1] / close[-4]) - 1) * 100 if len(close) > 4 else 0.0
    ret_6 = ((close[-1] / close[-7]) - 1) * 100 if len(close) > 7 else 0.0
    ret_10 = ((close[-1] / close[-11]) - 1) * 100 if len(close) > 11 else 0.0
    session_change = ((close[-1] / open_p[0]) - 1) * 100
    
    delta = np.diff(close[-15:])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain)
    avg_loss = np.mean(loss)
    rsi_14 = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
    
    feed_label = "OANDA CFD Live Feed" if "OANDA" in feed_source else "Yahoo Finance"

    # Multi-Horizon Trend Signals based on selected Model Variant (standalone vs rate_only vs macro_full)
    is_equity = name in ["S&P 500 (SPX)", "Nasdaq 100 (NDX)", "Russell 2000 (RUT)", "Nikkei 225 (NI225)"]
    has_macro = macro_dfs and "10Y Treasury (TNX)" in macro_dfs and macro_dfs["10Y Treasury (TNX)"] is not None

    if is_equity and model_variant == "rate_only" and has_macro:
        mh_data = generate_rate_only_signals(df, macro_dfs["10Y Treasury (TNX)"], equity_name=name, all_dfs=macro_dfs)
        mh_sigs = mh_data["signals"]
    elif is_equity and model_variant == "macro_full" and has_macro and "Brent Crude (BRENT)" in macro_dfs and "WTI Crude (WTI)" in macro_dfs:
        mh_data = generate_macro_full_signals(df, macro_dfs["10Y Treasury (TNX)"], macro_dfs["Brent Crude (BRENT)"], macro_dfs["WTI Crude (WTI)"], equity_name=name, all_dfs=macro_dfs)
        mh_sigs = mh_data["signals"]
    else:
        mh_data = generate_multi_horizon_signals(df, asset_name=name)
        mh_sigs = mh_data["signals"]

    sig_1m = mh_sigs["forward_1m"]
    sig_10m = mh_sigs["forward_10m"]
    sig_30m = mh_sigs["forward_30m"]
    sig_1h = mh_sigs["forward_1h"]

    act_1m, badge_1m, color_1m = format_horizon_badge(sig_1m["action"])
    act_10m, badge_10m, color_10m = format_horizon_badge(sig_10m["action"])
    act_30m, badge_30m, color_30m = format_horizon_badge(sig_30m["action"])
    act_1h, badge_1h, color_1h = format_horizon_badge(sig_1h["action"])

    jev_choice = sig_10m["action"]
    prob_up = sig_10m["prob_up"]
    confidence = sig_10m["confidence"]
    is_live_jev = mh_sigs.get("is_live_jev", False)
    state_str = mh_data.get("state_prompt", f"Asset: {name}")

    if "LONG" in jev_choice or "BUY" in jev_choice:
        action = "BUY / LONG"
        badge_class = "card-buy"
        color = "#00ff88"
    elif "SHORT" in jev_choice or "SELL" in jev_choice:
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
        "ret_10": ret_10,
        "state_str": state_str,
        "feed_source": feed_source,
        "timestamp": df.index[-1].strftime("%H:%M:%S UTC") if isinstance(df.index, pd.DatetimeIndex) else "NOW",
        "multi_horizon": {
            "model_type": mh_sigs.get("model_type", "STANDALONE"),
            "alignment": mh_sigs.get("alignment", "NEUTRAL"),
            "average_prob_up": mh_sigs.get("average_prob_up", round(prob_up, 4)),
            "forward_1m": {
                "horizon": "1min forward (1 bar)",
                "action": act_1m,
                "raw_action": sig_1m["action"],
                "prob_up": round(sig_1m["prob_up"], 4),
                "confidence": sig_1m["confidence"],
                "badge_class": badge_1m,
                "color": color_1m
            },
            "forward_10m": {
                "horizon": "10min forward (10 bars)",
                "action": act_10m,
                "raw_action": sig_10m["action"],
                "prob_up": round(sig_10m["prob_up"], 4),
                "confidence": sig_10m["confidence"],
                "badge_class": badge_10m,
                "color": color_10m
            },
            "forward_30m": {
                "horizon": "30min forward (30 bars)",
                "action": act_30m,
                "raw_action": sig_30m["action"],
                "prob_up": round(sig_30m["prob_up"], 4),
                "confidence": sig_30m["confidence"],
                "badge_class": badge_30m,
                "color": color_30m
            },
            "forward_1h": {
                "horizon": "1h forward (60 bars)",
                "action": act_1h,
                "raw_action": sig_1h["action"],
                "prob_up": round(sig_1h["prob_up"], 4),
                "confidence": sig_1h["confidence"],
                "badge_class": badge_1h,
                "color": color_1h
            },
            # Backward-compatibility aliases
            "forward_5m": {
                "horizon": "1min forward (1 bar)",
                "action": act_1m,
                "raw_action": sig_1m["action"],
                "prob_up": round(sig_1m["prob_up"], 4),
                "confidence": sig_1m["confidence"],
                "badge_class": badge_1m,
                "color": color_1m
            },
            "forward_15m": {
                "horizon": "30min forward (30 bars)",
                "action": act_30m,
                "raw_action": sig_30m["action"],
                "prob_up": round(sig_30m["prob_up"], 4),
                "confidence": sig_30m["confidence"],
                "badge_class": badge_30m,
                "color": color_30m
            }
        }
    }

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🌐 Global Multi-Index Momentum Radar")
    st.caption("Nikkei 225 • S&P 500 • Nasdaq 100 • Russell 2000 • 10Y Treasury • WTI & Brent Crude — Powered by TypeSafe Jev System One")
with header_col2:
    st.write("")
    st.markdown('<div style="text-align: right;"><span class="status-badge">🟢 JEV MODEL: ONLINE (jev-1.14.0)</span></div>', unsafe_allow_html=True)

# Top Notice if OANDA key is not yet provided
effective_key = oanda_key_input.strip() or os.getenv("OANDA_API_KEY", "")
if is_oanda_selected and not effective_key:
    st.warning("⚠️ **OANDA API Token Not Entered**: You selected OANDA Zero-Lag Feed, but no API key is provided yet. The radar is temporarily falling back to Yahoo Finance (which has a 15-minute delay). **Enter your OANDA API Key in the left sidebar** to unlock instant zero-delay 24/5 streaming.")

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 1, 1])

with ctrl_col1:
    timeframe = st.selectbox(
        "Timeframe Horizon",
        ["1m (1-Minute High-Frequency Micro-Scalp)", "5m (5-Minute Intraday Scalp)", "1h (1-Hour Tactical Swing)", "1d (Daily Holding)"],
        index=0
    )
    if "1m" in timeframe:
        tf_code = "1m"
    elif "5m" in timeframe:
        tf_code = "5m"
    elif "1h" in timeframe:
        tf_code = "1h"
    else:
        tf_code = "1d"

with ctrl_col2:
    model_choice = st.selectbox(
        "Cross-Asset Model Engine",
        [
            "Model A: Full Macro (10Y Yield + Brent + WTI)",
            "Model B: Rate-Only (10Y Yield ONLY, excl Oil)",
            "Standalone Microstructure (Single-Asset Only)"
        ],
        index=0
    )
    if "Model A" in model_choice:
        model_var = "macro_full"
    elif "Model B" in model_choice:
        model_var = "rate_only"
    else:
        model_var = "standalone"

with ctrl_col3:
    st.write("")
    st.write("")
    st.button("🔄 Refresh Radar Now", use_container_width=True)

with ctrl_col4:
    auto_refresh = st.checkbox("Auto-refresh (every 10s)", value=False)

# Fetch raw data for all assets first to enable cross-asset conditioning
raw_dfs = {}
feed_sources = {}
for name, meta in ASSETS.items():
    try:
        df, feed_src = fetch_asset_data_unified(name, meta, tf_code)
        if df is not None and len(df) >= 15:
            raw_dfs[name] = df
            feed_sources[name] = feed_src
    except Exception as e:
        st.warning(f"Error fetching {name}: {e}")

# Compute signals with selected model_var (Model A vs Model B vs Standalone)
results = {}
for name, meta in ASSETS.items():
    if name in raw_dfs:
        df = raw_dfs[name]
        feed_src = feed_sources.get(name, "OANDA")
        sig = analyze_asset(df, name=name, timeframe=tf_code, feed_source=feed_src, model_variant=model_var, macro_dfs=raw_dfs)
        if sig:
            results[name] = sig

# ---------------------------------------------------------
# 1. Multi-Index Cards Grid
# ---------------------------------------------------------
st.subheader("⚡ Live Decision Overview (TypeSafe Jev)")

asset_keys = list(ASSETS.keys())
row1_keys = asset_keys[:4]
row2_keys = asset_keys[4:]

# Row 1
cols1 = st.columns(4)
for i, name in enumerate(row1_keys):
    meta = ASSETS[name]
    sig = results.get(name)
    with cols1[i]:
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

# Row 2
cols2 = st.columns(len(row2_keys))
for i, name in enumerate(row2_keys):
    meta = ASSETS[name]
    sig = results.get(name)
    with cols2[i]:
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
        "Instrument": f"{ASSETS[name]['flag']} {name}",
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

# Multi-Horizon Forward Signal Matrix (1m, 10m, 30m, 1h Forward from 1m input)
st.subheader("⏱️ Multi-Horizon Forward Trend Signals (1m Input Interval)")
st.caption("Predictive directional trend classification and continuation probabilities for 1-min (1 bar), 10-min (10 bars), 30-min (30 bars), and 1-hour (60 bars) forward horizons.")

mh_matrix_data = []
for name, sig in results.items():
    mh = sig.get("multi_horizon", {})
    s1 = mh.get("forward_1m", mh.get("forward_5m", {}))
    s10 = mh.get("forward_10m", {})
    s30 = mh.get("forward_30m", mh.get("forward_15m", {}))
    s1h = mh.get("forward_1h", {})
    mh_matrix_data.append({
        "Instrument": f"{ASSETS[name]['flag']} {name}",
        "1m Forward Signal": f"{s1.get('action', 'NEUTRAL')} ({s1.get('prob_up', 0.5)*100:.1f}%)",
        "10m Forward Signal": f"{s10.get('action', 'NEUTRAL')} ({s10.get('prob_up', 0.5)*100:.1f}%)",
        "30m Forward Signal": f"{s30.get('action', 'NEUTRAL')} ({s30.get('prob_up', 0.5)*100:.1f}%)",
        "1h Forward Signal": f"{s1h.get('action', 'NEUTRAL')} ({s1h.get('prob_up', 0.5)*100:.1f}%)",
        "Multi-Horizon Alignment": mh.get("alignment", "NEUTRAL"),
        "Mean Prob P(Up)": f"{mh.get('average_prob_up', 0.5)*100:.1f}%"
    })
mh_matrix_df = pd.DataFrame(mh_matrix_data)
st.dataframe(mh_matrix_df, use_container_width=True, hide_index=True)

# Cross-Market Intelligence (10-Minute Horizon)
if "S&P 500 (SPX)" in results and "Nasdaq 100 (NDX)" in results and "Russell 2000 (RUT)" in results and "Nikkei 225 (NI225)" in results:
    sp = results["S&P 500 (SPX)"]
    ndx = results["Nasdaq 100 (NDX)"]
    rut = results["Russell 2000 (RUT)"]
    nik = results["Nikkei 225 (NI225)"]
    
    tech_spread = ndx["ret_10"] - sp["ret_10"]
    small_spread = rut["ret_10"] - sp["ret_10"]
    
    insights = []
    if tech_spread > 0.10:
        insights.append("🟢 **Tech Outperformance (NDX > SPX)**: Tech sector CFD leadership is driving 10-minute momentum.")
    elif tech_spread < -0.10:
        insights.append("🟡 **Tech Drag (NDX < SPX)**: Tech CFD sector lagging broader index momentum over 10m.")
        
    if small_spread > 0.15:
        insights.append("🚀 **Broad Risk-On (RUT > SPX)**: Small caps showing high-beta CFD participation.")
    elif small_spread < -0.15:
        insights.append("⚠️ **Defensive Posture (RUT < SPX)**: Small caps underperforming over last 10m; watch for false large-cap breakouts.")
        
    if nik["prob_up"] > 0.55:
        insights.append("🇯🇵 **Nikkei Momentum**: Asian session trading with upside momentum bias.")

    if "WTI Crude (WTI)" in results and "Brent Crude (BRENT)" in results:
        wti = results["WTI Crude (WTI)"]
        brent = results["Brent Crude (BRENT)"]
        spread = brent["price"] - wti["price"]
        if brent["ret_10"] > 0.20 and wti["ret_10"] > 0.20:
            insights.append(f"🛢️ **Energy Surge (Crude Momentum Up)**: WTI (${wti['price']:.2f}) & Brent (${brent['price']:.2f}) surging. Brent-WTI Spread: ${spread:.2f}. Watch for headline inflation and yields pressure.")
        elif brent["ret_10"] < -0.20 and wti["ret_10"] < -0.20:
            insights.append(f"🌊 **Crude Deflationary Easing**: Oil selling off (WTI {wti['ret_10']:+.2f}%, Brent {brent['ret_10']:+.2f}%). Easing energy cost pressure across global markets.")
        
    if insights:
        st.info("💡 **Cross-Asset Intelligence:**\n\n" + "\n\n".join(insights))

st.divider()

# ---------------------------------------------------------
# 3. Interactive Candlestick Chart
# ---------------------------------------------------------
selected_asset = st.selectbox("Select Asset for Deep-Dive Chart", list(ASSETS.keys()), index=0)
if selected_asset in raw_dfs:
    chart_df = raw_dfs[selected_asset].iloc[-50:]
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
    feed_title_tag = "OANDA CFD (Zero-Lag)" if "OANDA" in sel_sig["feed_source"] else "Yahoo (15m Delayed)"
    fig.update_layout(
        title=f"{ASSETS[selected_asset]['flag']} {selected_asset} [{feed_title_tag}] - {timeframe}",
        height=430,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

if auto_refresh:
    import time
    time.sleep(10)
    st.rerun()
