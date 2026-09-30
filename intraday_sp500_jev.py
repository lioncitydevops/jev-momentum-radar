"""
Intraday (5-Minute / 1-Hour) S&P 500 Momentum Engine for Jev System One
Author: Quantitative Trader & Mathematician
"""

import json
import math
import numpy as np
import pandas as pd
import requests

def fetch_intraday_data(symbol: str = "SPY", interval: str = "5m", range_str: str = "5d") -> pd.DataFrame:
    """Fetches intraday bars (5m or 1h) for SPY."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_str}"
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
    
    # Convert index to US/Eastern timezone
    df.index = df.index.tz_convert("America/New_York")
    df = df.dropna()
    return df

def compute_intraday_momentum_state(df: pd.DataFrame) -> dict:
    """
    Computes microstructure and intraday momentum features:
    - Anchored Session VWAP and VWAP Z-score
    - Intraday Multi-Bar Momentum (3-bar = 15m, 6-bar = 30m, 12-bar = 60m)
    - Relative Volume (RVOL) vs recent average
    - Time-of-Day Regime (Opening Drive, Midday Lull, Power Hour)
    """
    data = df.copy()
    
    # Identify today's session bars (anchored to current trading day)
    today = data.index[-1].date()
    today_mask = data.index.date == today
    session_df = data.loc[today_mask]
    
    if len(session_df) == 0:
        session_df = data.iloc[-78:] # fallback to last 78 bars (~1 session)
        
    # Anchored Session VWAP
    cum_vol = session_df["Volume"].cumsum()
    cum_vol_price = (session_df["Volume"] * (session_df["High"] + session_df["Low"] + session_df["Close"]) / 3.0).cumsum()
    session_vwap = cum_vol_price / (cum_vol + 1e-9)
    current_vwap = session_vwap.iloc[-1]
    
    # Intraday ATR / Standard deviation of 5m bars
    close = data["Close"].values
    high = data["High"].values
    low = data["Low"].values
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr_14 = float(np.mean(tr[-14:]))
    
    # VWAP Z-score (Distance in ATR units)
    vwap_dist = (close[-1] - current_vwap)
    vwap_z = float(np.round(vwap_dist / (atr_14 + 1e-9), 2))
    
    # Micro-momentum (Returns over last 15m, 30m, 60m)
    # At 5m bars: 3 bars = 15m, 6 bars = 30m, 12 bars = 60m
    ret_15m = float(np.round((close[-1] / close[-4] - 1) * 100, 3)) if len(close) > 4 else 0.0
    ret_30m = float(np.round((close[-1] / close[-7] - 1) * 100, 3)) if len(close) > 7 else 0.0
    ret_60m = float(np.round((close[-1] / close[-13] - 1) * 100, 3)) if len(close) > 13 else 0.0
    session_open = session_df["Open"].iloc[0]
    session_ret = float(np.round((close[-1] / session_open - 1) * 100, 3))
    
    # Relative Volume (RVOL)
    recent_vol = data["Volume"].iloc[-1]
    avg_vol = data["Volume"].iloc[-20:].mean()
    rvol = float(np.round(recent_vol / (avg_vol + 1e-9), 2))
    
    # Time of Day Classification (US Eastern)
    current_time = data.index[-1].time()
    time_str = current_time.strftime("%H:%M")
    
    if current_time < pd.to_datetime("10:30").time():
        regime = "OPENING_DRIVE (High momentum volatility & price discovery)"
    elif current_time < pd.to_datetime("14:30").time():
        regime = "MIDDAY_LULL (Mean-reverting chop, lower institutional volume)"
    elif current_time < pd.to_datetime("15:30").time():
        regime = "AFTERNOON_POSITIONING (Trend resumption or pre-close hedging)"
    else:
        regime = "POWER_HOUR (Aggressive market-on-close rebalancing & momentum surge)"
        
    state_prompt = (
        f"Asset: S&P 500 (SPY 5-Minute Intraday Bar)\n"
        f"Bar Timestamp (ET): {data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Latest 5m Close: ${close[-1]:.2f}\n"
        f"Intraday Session VWAP: ${current_vwap:.2f} (Distance: {vwap_z:+.2f} ATRs)\n"
        f"Micro-Momentum: 15-min={ret_15m:+.2f}%, 30-min={ret_30m:+.2f}%, 60-min={ret_60m:+.2f}%\n"
        f"Full Session Return: {session_ret:+.2f}%\n"
        f"Relative Volume (RVOL): {rvol:.2f}x (Current bar volume vs 20-bar average)\n"
        f"5-Minute ATR: ${atr_14:.2f}\n"
        f"Intraday Time Regime: {regime}"
    )
    
    return {
        "state_str": state_prompt,
        "metrics": {
            "close": close[-1],
            "vwap": current_vwap,
            "vwap_z": vwap_z,
            "ret_15m": ret_15m,
            "ret_30m": ret_30m,
            "ret_60m": ret_60m,
            "rvol": rvol,
            "time_regime": regime
        }
    }

if __name__ == "__main__":
    print("=== Fetching Real 5-Minute S&P 500 Intraday Data ===")
    df = fetch_intraday_data("SPY", interval="5m", range_str="5d")
    result = compute_intraday_momentum_state(df)
    
    print("\n" + "=" * 65)
    print("5-MINUTE INTRADAY STATE FOR JEV SYSTEM ONE:")
    print("=" * 65)
    print(result["state_str"])
    print("=" * 65)
