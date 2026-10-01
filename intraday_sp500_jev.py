"""
Intraday (5-Minute / 1-Hour) S&P 500 Momentum Engine for Jev System One
Author: Quantitative Trader & Mathematician
"""

import json
import math
import numpy as np
import pandas as pd
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def fetch_intraday_data(symbol: str = "SPX500_USD", interval: str = "1m", range_str: str = "2d") -> pd.DataFrame:
    """
    Fetches intraday bars (1m, 5m, or 1h) for S&P 500.
    Prioritizes OANDA real-time 24/5 CFD feed (SPX500_USD) if OANDA_API_KEY is configured.
    Falls back to Yahoo Finance (SPY).
    """
    oanda_key = os.getenv("OANDA_API_KEY", "").strip()
    if oanda_key:
        try:
            from oanda_feed import fetch_oanda_candles
            print(f"[DATA FEED] Fetching live 24/5 zero-delay 1-minute bars from OANDA ({symbol})...")
            df = fetch_oanda_candles(instrument=symbol, timeframe=interval, count=120, api_key=oanda_key)
            df.index = df.index.tz_convert("America/New_York")
            return df
        except Exception as e:
            print(f"[DATA FEED WARNING] OANDA fetch failed ({e}). Falling back to Yahoo Finance...")

    # Fallback to Yahoo Finance
    print("[DATA FEED] Fetching bars from Yahoo Finance (15m delay)...")
    yahoo_symbol = "SPY" if symbol in ["SPX500_USD", "SPX"] else symbol
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={interval}&range={range_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()["chart"]["result"][0]
    
    timestamps = data["timestamp"]
    quote = data["indicators"]["quote"][0]
    
    df = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote["volume"],
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))
    
    df.index = df.index.tz_convert("America/New_York")
    df = df.dropna()
    return df

def compute_intraday_momentum_state(df: pd.DataFrame, interval: str = "1m") -> dict:
    """
    Computes microstructure and intraday momentum features:
    - Anchored Session VWAP and VWAP Z-score
    - Multi-Bar Momentum (3-bar, 5-bar, 15-bar for 1m; or 15m, 30m, 60m for 5m)
    - Relative Volume (RVOL) vs recent average
    - Time-of-Day Regime (Opening Drive, Midday Lull, Power Hour)
    """
    data = df.copy()
    
    # Identify today's session bars (anchored to current trading day)
    today = data.index[-1].date()
    today_mask = data.index.date == today
    session_df = data.loc[today_mask]
    
    if len(session_df) == 0:
        session_df = data.iloc[-60:] # fallback to last 60 bars
        
    # Anchored Session VWAP
    cum_vol = session_df["Volume"].cumsum()
    cum_vol_price = (session_df["Volume"] * (session_df["High"] + session_df["Low"] + session_df["Close"]) / 3.0).cumsum()
    session_vwap = cum_vol_price / (cum_vol + 1e-9)
    current_vwap = session_vwap.iloc[-1]
    
    # Intraday ATR
    close = data["Close"].values
    high = data["High"].values
    low = data["Low"].values
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr_14 = float(np.mean(tr[-14:]))
    
    # VWAP Z-score (Distance in ATR units)
    vwap_dist = (close[-1] - current_vwap)
    vwap_z = float(np.round(vwap_dist / (atr_14 + 1e-9), 2))
    
    # Micro-momentum
    if interval == "1m":
        ret_1 = float(np.round((close[-1] / close[-4] - 1) * 100, 3)) if len(close) > 4 else 0.0 # 3-min
        ret_2 = float(np.round((close[-1] / close[-6] - 1) * 100, 3)) if len(close) > 6 else 0.0 # 5-min
        ret_3 = float(np.round((close[-1] / close[-16] - 1) * 100, 3)) if len(close) > 16 else 0.0 # 15-min
        m_label = f"3-min={ret_1:+.2f}%, 5-min={ret_2:+.2f}%, 15-min={ret_3:+.2f}%"
    else:
        ret_1 = float(np.round((close[-1] / close[-4] - 1) * 100, 3)) if len(close) > 4 else 0.0 # 15-min
        ret_2 = float(np.round((close[-1] / close[-7] - 1) * 100, 3)) if len(close) > 7 else 0.0 # 30-min
        ret_3 = float(np.round((close[-1] / close[-13] - 1) * 100, 3)) if len(close) > 13 else 0.0 # 60-min
        m_label = f"15-min={ret_1:+.2f}%, 30-min={ret_2:+.2f}%, 60-min={ret_3:+.2f}%"

    session_open = session_df["Open"].iloc[0]
    session_ret = float(np.round((close[-1] / session_open - 1) * 100, 3))
    
    # Relative Volume (RVOL)
    recent_vol = data["Volume"].iloc[-1]
    avg_vol = data["Volume"].iloc[-20:].mean()
    rvol = float(np.round(recent_vol / (avg_vol + 1e-9), 2))
    
    # Time of Day Classification (US Eastern)
    current_time = data.index[-1].time()
    
    if current_time < pd.to_datetime("10:30").time():
        regime = "OPENING_DRIVE (High momentum volatility & price discovery)"
    elif current_time < pd.to_datetime("14:30").time():
        regime = "MIDDAY_LULL (Mean-reverting chop, lower institutional volume)"
    elif current_time < pd.to_datetime("15:30").time():
        regime = "AFTERNOON_POSITIONING (Trend resumption or pre-close hedging)"
    else:
        regime = "POWER_HOUR (Aggressive market-on-close rebalancing & momentum surge)"
        
    state_prompt = (
        f"Asset: S&P 500 CFD ({interval.upper()} Intraday Bar - OANDA Live Feed)\n"
        f"Forecasting Target: Next 10 Minutes Movement (10 bars forward)\n"
        f"Bar Timestamp (ET): {data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Latest {interval} Close: ${close[-1]:.2f}\n"
        f"Intraday Session VWAP: ${current_vwap:.2f} (Distance: {vwap_z:+.2f} ATRs)\n"
        f"Micro-Momentum: {m_label}\n"
        f"Full Session Return: {session_ret:+.2f}%\n"
        f"Relative Volume (RVOL): {rvol:.2f}x (Current bar volume vs 20-bar average)\n"
        f"{interval.upper()} ATR: ${atr_14:.2f}\n"
        f"Intraday Time Regime: {regime}"
    )
    
    return {
        "state_str": state_prompt,
        "metrics": {
            "close": close[-1],
            "vwap": current_vwap,
            "vwap_z": vwap_z,
            "ret_fast": ret_1,
            "ret_mid": ret_2,
            "ret_slow": ret_3,
            "rvol": rvol,
            "time_regime": regime
        }
    }

if __name__ == "__main__":
    print("=== Fetching Real 1-Minute S&P 500 Intraday Data ===")
    df = fetch_intraday_data("SPX500_USD", interval="1m", range_str="2d")
    result = compute_intraday_momentum_state(df, interval="1m")
    
    print("\n" + "=" * 65)
    print("1-MINUTE HIGH-FREQUENCY INTRADAY STATE FOR JEV SYSTEM ONE:")
    print("=" * 65)
    print(result["state_str"])
    print("=" * 65)
