"""
Live S&P 500 Momentum Calculator & TypeSafe AI Prompt Generator
Fetches live SPY and VIX market data and prints the exact State and Questions
ready to paste directly into https://console.typesafe.ai/home
"""

import json
import math
import numpy as np
import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

def fetch_chart(symbol: str, range_str: str = "1mo") -> pd.DataFrame:
    oanda_key = os.getenv("OANDA_API_KEY", "").strip()
    if oanda_key and symbol in ["SPY", "SPX", "SPX500_USD"]:
        try:
            from oanda_feed import fetch_oanda_candles
            print("[DATA FEED] Using live 24/5 OANDA CFD feed (SPX500_USD)...")
            return fetch_oanda_candles("SPX500_USD", timeframe="1d", count=30, api_key=oanda_key)
        except Exception as e:
            print(f"[DATA FEED WARNING] OANDA fetch failed ({e}). Falling back to Yahoo...")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_str}"
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
    }, index=pd.to_datetime(timestamps, unit="s"))
    df = df.dropna()
    return df

def generate_live_prompt():
    print("Fetching live market data for S&P 500 (OANDA/SPY) and VIX...")
    spy_df = fetch_chart("SPY", "1mo")
    vix_df = fetch_chart("%5EVIX", "5d")
    
    close = spy_df["Close"].values
    high = spy_df["High"].values
    low = spy_df["Low"].values
    open_p = spy_df["Open"].values
    current_vix = vix_df["Close"].iloc[-1]
    
    # 20-day Garman-Klass Volatility
    log_hl = np.log(high[-20:] / low[-20:])
    log_co = np.log(close[-20:] / open_p[-20:])
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    daily_vol = np.sqrt(np.mean(gk_var))
    annual_vol = daily_vol * np.sqrt(252) * 100
    
    # Momentum Z-Scores (1-day, 3-day, 5-day, 10-day)
    horizons = [1, 3, 5, 10]
    z_scores = {}
    for h in horizons:
        ret = np.log(close[-1] / close[-1 - h])
        z_scores[h] = ret / (daily_vol * np.sqrt(h))
    
    # 14-day RSI
    delta = np.diff(close[-15:])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain)
    avg_loss = np.mean(loss)
    rsi_14 = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
    
    # Distance from 20-day EMA
    ema_20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
    pct_from_ema20 = ((close[-1] - ema_20) / ema_20) * 100
    
    # Close location within the day's High-Low range (0 to 1)
    day_range = high[-1] - low[-1]
    close_loc = (close[-1] - low[-1]) / (day_range + 1e-9)
    
    state_text = (
        f"Asset: S&P 500 Index CFD (SPX)\n"
        f"Latest Close Price: ${close[-1]:.2f}\n"
        f"1-Day Normalized Momentum: {z_scores[1]:+.2f} standard deviations\n"
        f"3-Day Normalized Momentum: {z_scores[3]:+.2f} standard deviations\n"
        f"5-Day Normalized Momentum: {z_scores[5]:+.2f} standard deviations\n"
        f"10-Day Normalized Momentum: {z_scores[10]:+.2f} standard deviations\n"
        f"14-Day RSI: {rsi_14:.1f}\n"
        f"Distance from 20-Day EMA: {pct_from_ema20:+.2f}%\n"
        f"20-Day Realized Daily Volatility: {daily_vol*100:.2f}% (Annualized: {annual_vol:.1f}%)\n"
        f"VIX Volatility Index: {current_vix:.2f}\n"
        f"Day Range Close Location (0.0 = Low of Day, 1.0 = High of Day): {close_loc:.2f}"
    )
    
    return state_text

if __name__ == "__main__":
    prompt = generate_live_prompt()
    print("\n" + "="*60)
    print("COPY THIS STATE INTO https://console.typesafe.ai/home PLAYGROUND:")
    print("="*60)
    print(prompt)
    print("="*60)
