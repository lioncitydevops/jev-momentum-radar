import os
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

ASSETS = {
    "S&P 500": {"symbol": "SPY", "flag": "🇺🇸", "desc": "US Large-Cap Benchmark"},
    "Nasdaq 100": {"symbol": "QQQ", "flag": "💻", "desc": "US Tech & Growth Leaders"},
    "Russell 2000": {"symbol": "IWM", "flag": "🚀", "desc": "US Small-Cap Risk-On"},
    "Nikkei 225": {"symbol": "^N225", "flag": "🇯🇵", "desc": "Japan Benchmark Index"}
}

def fetch_asset_data(symbol: str, timeframe: str = "5m"):
    range_str = "5d" if timeframe == "5m" else ("1mo" if timeframe == "1h" else "3mo")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={timeframe}&range={range_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=8)
    data = resp.json()["chart"]["result"][0]
    
    timestamps = data["timestamp"]
    quote = data["indicators"]["quote"][0]
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote.get("volume", [0] * len(timestamps)),
    })
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df

def call_jev_api(state_str: str) -> dict:
    if not TYPESAFE_API_KEY:
        raise ValueError("No TYPESAFE_API_KEY configured")
        
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
    
    try:
        jev_res = call_jev_api(state_str)
        jev_choice = jev_res["choice"]
        prob_up = float(jev_res["prob_up"])
        confidence = float(jev_res["confidence"])
        is_live_jev = True
    except Exception:
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

    # Last 45 candles for charting
    last_df = df.iloc[-45:]
    candles = []
    for _, row in last_df.iterrows():
        candles.append({
            "t": int(row["Timestamp"]),
            "o": round(float(row["Open"]), 2),
            "h": round(float(row["High"]), 2),
            "l": round(float(row["Low"]), 2),
            "c": round(float(row["Close"]), 2),
            "v": int(row["Volume"]) if not np.isnan(row["Volume"]) else 0
        })

    return {
        "name": name,
        "symbol": ASSETS[name]["symbol"],
        "flag": ASSETS[name]["flag"],
        "desc": ASSETS[name]["desc"],
        "action": action,
        "badge_class": badge_class,
        "color": color,
        "prob_up": round(prob_up, 4),
        "confidence": round(confidence, 4),
        "is_live_jev": is_live_jev,
        "price": round(float(close[-1]), 2),
        "session_change": round(float(session_change), 2),
        "vwap": round(float(vwap), 2),
        "vwap_z": round(float(vwap_z), 2),
        "rsi": round(float(rsi_14), 1),
        "ret_3": round(float(ret_3), 2),
        "ret_6": round(float(ret_6), 2),
        "state_str": state_str,
        "candles": candles
    }

def generate_insights(results):
    insights = []
    if len(results) >= 4 and "S&P 500" in results and "Nasdaq 100" in results and "Russell 2000" in results and "Nikkei 225" in results:
        sp = results["S&P 500"]
        ndx = results["Nasdaq 100"]
        rut = results["Russell 2000"]
        nik = results["Nikkei 225"]
        
        tech_spread = ndx["ret_6"] - sp["ret_6"]
        small_spread = rut["ret_6"] - sp["ret_6"]
        
        if tech_spread > 0.15:
            insights.append({"type": "bull", "text": "Tech Outperformance (QQQ > SPY): Tech leadership is driving index momentum."})
        elif tech_spread < -0.15:
            insights.append({"type": "warn", "text": "Tech Drag (QQQ < SPY): Tech sector lagging broader market."})
            
        if small_spread > 0.20:
            insights.append({"type": "bull", "text": "Broad Risk-On (IWM > SPY): Small caps showing high-beta participation."})
        elif small_spread < -0.20:
            insights.append({"type": "warn", "text": "Defensive Posture (IWM < SPY): Small caps underperforming; watch for false large-cap breakouts."})
            
        if nik["prob_up"] > 0.55:
            insights.append({"type": "info", "text": "Nikkei Momentum: Asian session trading with upside momentum bias."})
            
    return insights

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_url.query)
        tf_code = query.get("timeframe", ["5m"])[0]
        if tf_code not in ["5m", "1h", "1d"]:
            tf_code = "5m"

        results = {}
        for name, meta in ASSETS.items():
            try:
                df = fetch_asset_data(meta["symbol"], tf_code)
                sig = analyze_asset(df, name=name, timeframe=tf_code)
                if sig:
                    results[name] = sig
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        insights = generate_insights(results)
        response_data = {
            "status": "success",
            "timeframe": tf_code,
            "results": results,
            "insights": insights
        }

        body = json.dumps(response_data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    # Test script locally
    import sys
    print("Testing radar fetch locally...")
    for name, meta in ASSETS.items():
        df = fetch_asset_data(meta["symbol"], "5m")
        sig = analyze_asset(df, name=name, timeframe="5m")
        print(f"{name}: Price=${sig['price']}, Action={sig['action']}, Prob={sig['prob_up']}, JevLive={sig['is_live_jev']}")
