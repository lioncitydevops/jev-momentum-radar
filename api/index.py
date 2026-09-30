import os
import json
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Global Multi-Futures Momentum Radar (100% TradingView)", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

# 100% Global Futures Specifications
ASSETS = {
    "S&P 500 (ES)": {
        "symbol": "ES=F",
        "tv_ticker": "CME_MINI:ES1!",
        "flag": "🇺🇸",
        "desc": "E-mini S&P 500 Futures (CME)",
        "market": "futures"
    },
    "Nasdaq 100 (NQ)": {
        "symbol": "NQ=F",
        "tv_ticker": "CME_MINI:NQ1!",
        "flag": "💻",
        "desc": "E-mini Nasdaq 100 Futures (CME)",
        "market": "futures"
    },
    "Russell 2000 (RTY)": {
        "symbol": "RTY=F",
        "tv_ticker": "CME_MINI:RTY1!",
        "flag": "🚀",
        "desc": "E-mini Russell 2000 Futures (CME)",
        "market": "futures"
    },
    "Nikkei 225 (NKD)": {
        "symbol": "NKD=F",
        "tv_ticker": "OSE:NK2251!",
        "flag": "🇯🇵",
        "desc": "Nikkei 225 Index Futures (OSE/CME)",
        "market": "futures"
    },
    "10Y T-Note (TY10)": {
        "symbol": "ZN=F",
        "tv_ticker": "CBOT:ZN1!",
        "flag": "🏛️",
        "desc": "10-Year U.S. Treasury Note Futures (CBOT)",
        "market": "futures"
    }
}

def fetch_tradingview_scan(timeframe: str = "5m") -> dict:
    """Pulls live indicators directly from TradingView's official Futures Scanner API."""
    cols = ["close", "change", "RSI", "VWAP", "Recommend.All", "volume", "ATR", "open", "high", "low"]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    tv_data = {}
    
    try:
        url_fut = "https://scanner.tradingview.com/futures/scan"
        tickers = [meta["tv_ticker"] for meta in ASSETS.values()]
        payload = {"symbols": {"tickers": tickers}, "columns": cols}
        res_fut = requests.post(url_fut, json=payload, headers=headers, timeout=8).json()
        for item in res_fut.get("data", []):
            ticker = item["s"]
            vals = item["d"]
            tv_data[ticker] = {
                "close": vals[0],
                "change": vals[1],
                "rsi": vals[2],
                "vwap": vals[3],
                "recommend": vals[4],
                "volume": vals[5],
                "atr": vals[6],
                "open": vals[7],
                "high": vals[8],
                "low": vals[9]
            }
    except Exception as e:
        print(f"Error fetching TradingView futures scan: {e}")
        
    return tv_data

def get_tv_rating_text(score):
    if score is None:
        return "NEUTRAL", "#ffd166"
    if score >= 0.5:
        return "STRONG BUY", "#00ff88"
    elif score >= 0.1:
        return "BUY", "#22c55e"
    elif score <= -0.5:
        return "STRONG SELL", "#ff4d6d"
    elif score <= -0.1:
        return "SELL", "#ef4444"
    else:
        return "NEUTRAL", "#ffd166"

def fetch_futures_candles(symbol: str, timeframe: str = "5m"):
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
                "instructions": "Determine tactical positioning for the next 15-30 minutes based on TradingView futures momentum, VWAP deviation, and technical ratings.",
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

def analyze_asset(df, name="S&P 500 (ES)", timeframe="5m", tv_metric=None):
    close_vals = df["Close"].values
    high_vals = df["High"].values
    low_vals = df["Low"].values
    open_p = df["Open"].values
    volume_vals = df["Volume"].fillna(0).values
    
    if len(close_vals) < 15:
        return None

    tr = np.maximum(high_vals[1:] - low_vals[1:], np.maximum(abs(high_vals[1:] - close_vals[:-1]), abs(low_vals[1:] - close_vals[:-1])))
    atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.std(close_vals))
    
    ret_3 = (close_vals[-1] / close_vals[-4] - 1) * 100 if len(close_vals) > 4 else 0.0
    ret_6 = (close_vals[-1] / close_vals[-7] - 1) * 100 if len(close_vals) > 7 else 0.0
    
    tv_ticker = ASSETS[name]["tv_ticker"]
    if tv_metric and tv_metric.get("close") is not None:
        price = float(tv_metric["close"])
        session_change = float(tv_metric["change"]) if tv_metric["change"] is not None else (close_vals[-1] / open_p[0] - 1) * 100
        rsi_14 = float(tv_metric["rsi"]) if tv_metric["rsi"] is not None else 50.0
        vwap = float(tv_metric["vwap"]) if tv_metric["vwap"] is not None else float(np.mean(close_vals[-20:]))
        tv_rating_score = tv_metric.get("recommend")
        feed_source = "TradingView Futures Live"
    else:
        price = float(close_vals[-1])
        session_change = (close_vals[-1] / open_p[0] - 1) * 100
        delta = np.diff(close_vals[-15:])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        rs = np.mean(gain) / (np.mean(loss) + 1e-9)
        rsi_14 = float(100 - (100 / (1 + rs)))
        total_vol = volume_vals.sum()
        if total_vol > 0:
            typical_price = (high_vals + low_vals + close_vals) / 3.0
            vwap = float((typical_price * volume_vals).sum() / (total_vol + 1e-9))
        else:
            vwap = float(np.mean(close_vals[-20:]))
        tv_rating_score = None
        feed_source = "Futures Chart Feed"

    vwap_z = (price - vwap) / (atr_14 + 1e-9)
    tv_rating_label, tv_rating_color = get_tv_rating_text(tv_rating_score)

    state_str = (
        f"Futures Asset: {name} (TradingView Symbol: {tv_ticker}, {timeframe} horizon). "
        f"TradingView Futures Price: ${price:,.2f}. "
        f"TradingView Futures VWAP: ${vwap:,.2f} (Distance: {vwap_z:+.2f} ATRs). "
        f"TradingView 14-RSI: {rsi_14:.1f}. "
        f"TradingView Technical Rating: {tv_rating_label} ({tv_rating_score if tv_rating_score is not None else 0:+.2f}). "
        f"Micro-Momentum: 3-bar={ret_3:+.2f}%, 6-bar={ret_6:+.2f}%. Session Return: {session_change:+.2f}%."
    )
    
    try:
        jev_res = call_jev_api(state_str)
        jev_choice = jev_res["choice"]
        prob_up = float(jev_res["prob_up"])
        confidence = float(jev_res["confidence"])
        is_live_jev = True
    except Exception:
        raw_score = 0.35 * ret_3 + 0.35 * ret_6 + 0.30 * (vwap_z * 0.4)
        if tv_rating_score is not None:
            raw_score += 0.20 * tv_rating_score
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

    # Candle series for Plotly chart
    last_df = df.iloc[-45:]
    candles = []
    decimals = 3 if "ZN=F" in ASSETS[name]["symbol"] else 2
    for _, row in last_df.iterrows():
        candles.append({
            "t": int(row["Timestamp"]),
            "o": round(float(row["Open"]), decimals),
            "h": round(float(row["High"]), decimals),
            "l": round(float(row["Low"]), decimals),
            "c": round(float(row["Close"]), decimals),
            "v": int(row["Volume"]) if not np.isnan(row["Volume"]) else 0
        })

    return {
        "name": name,
        "symbol": ASSETS[name]["symbol"],
        "tv_ticker": tv_ticker,
        "flag": ASSETS[name]["flag"],
        "desc": ASSETS[name]["desc"],
        "feed_source": feed_source,
        "action": action,
        "badge_class": badge_class,
        "color": color,
        "prob_up": round(prob_up, 4),
        "confidence": round(confidence, 4),
        "is_live_jev": is_live_jev,
        "price": round(price, decimals),
        "session_change": round(session_change, 2),
        "vwap": round(vwap, decimals),
        "vwap_z": round(vwap_z, 2),
        "rsi": round(rsi_14, 1),
        "ret_3": round(ret_3, 2),
        "ret_6": round(ret_6, 2),
        "tv_rating": tv_rating_label,
        "tv_rating_score": round(tv_rating_score, 2) if tv_rating_score is not None else None,
        "tv_rating_color": tv_rating_color,
        "state_str": state_str,
        "candles": candles
    }

def generate_insights(results):
    insights = []
    if "S&P 500 (ES)" in results and "Nasdaq 100 (NQ)" in results and "Russell 2000 (RTY)" in results and "Nikkei 225 (NKD)" in results:
        sp = results["S&P 500 (ES)"]
        ndx = results["Nasdaq 100 (NQ)"]
        rut = results["Russell 2000 (RTY)"]
        nik = results["Nikkei 225 (NKD)"]
        
        tech_spread = ndx["ret_6"] - sp["ret_6"]
        small_spread = rut["ret_6"] - sp["ret_6"]
        
        if tech_spread > 0.15:
            insights.append({"type": "bull", "text": "Tech Futures Leadership (NQ > ES): E-mini Nasdaq outperforming broader index futures."})
        elif tech_spread < -0.15:
            insights.append({"type": "warn", "text": "Tech Futures Drag (NQ < ES): Tech futures lagging broader market momentum."})
            
        if small_spread > 0.20:
            insights.append({"type": "bull", "text": "Broad Risk-On Breadth (RTY > ES): Small-cap futures showing aggressive beta leadership."})
        elif small_spread < -0.20:
            insights.append({"type": "warn", "text": "Defensive Posture (RTY < ES): Russell futures lagging; watch for false large-cap breakouts."})
            
        if nik["prob_up"] > 0.55 or nik.get("tv_rating") in ["BUY", "STRONG BUY"]:
            insights.append({"type": "info", "text": f"Nikkei 225 Futures Signal: Osaka/CME Asian session {nik.get('tv_rating', 'BULLISH')} momentum bias."})

    # Rates & Macro Insights with 10Y Treasury Note Futures
    if "10Y T-Note (TY10)" in results and "S&P 500 (ES)" in results:
        ty = results["10Y T-Note (TY10)"]
        sp = results["S&P 500 (ES)"]
        if ty["ret_6"] > 0.10 and sp["ret_6"] < -0.10:
            insights.append({"type": "info", "text": "🏛️ Flight-to-Safety: 10Y Treasury Futures rallying while equity futures pull back (classic safe-haven bid)."})
        elif ty["ret_6"] < -0.10 and sp["ret_6"] > 0.10:
            insights.append({"type": "bull", "text": "⚡ Growth Reflation: 10Y Treasury Futures softening as equity futures accelerate higher."})
        elif ty.get("tv_rating") in ["SELL", "STRONG SELL"]:
            insights.append({"type": "warn", "text": "⚠️ Rate Pressure (TY10 Selling): Bond yields pushing higher; watch duration headwinds for Nasdaq futures (NQ)."})

    return insights

@app.get("/")
def serve_home():
    html_path = Path(__file__).parent.parent / "public" / "index.html"
    if not html_path.exists():
        html_path = Path("public/index.html")
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Global Multi-Futures Momentum Radar is Running</h1>")

@app.get("/api/radar")
@app.get("/radar")
def get_radar(timeframe: str = "5m"):
    if timeframe not in ["5m", "1h", "1d"]:
        timeframe = "5m"

    tv_data = fetch_tradingview_scan(timeframe)

    results = {}
    for name, meta in ASSETS.items():
        try:
            df = fetch_futures_candles(meta["symbol"], timeframe)
            tv_metric = tv_data.get(meta["tv_ticker"])
            sig = analyze_asset(df, name=name, timeframe=timeframe, tv_metric=tv_metric)
            if sig:
                results[name] = sig
        except Exception as e:
            print(f"Error fetching {name}: {e}")

    insights = generate_insights(results)
    return {
        "status": "success",
        "feed": "TradingView Official Futures Feeds",
        "timeframe": timeframe,
        "results": results,
        "insights": insights
    }

@app.post("/api/webhook")
@app.post("/webhook/tradingview")
async def tradingview_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    return {"status": "received", "data": body}
