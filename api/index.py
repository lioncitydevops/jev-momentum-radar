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

app = FastAPI(title="Global Multi-CFD Momentum Radar (100% TradingView)", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

# 100% Global CFD Specifications
ASSETS = {
    "S&P 500 (SPX)": {
        "symbol": "^GSPC",
        "oanda": "SPX500_USD",
        "tv_ticker": "SP:SPX",
        "flag": "🇺🇸",
        "desc": "S&P 500 Index / CFD",
        "market": "cfd"
    },
    "Nasdaq 100 (NDX)": {
        "symbol": "^NDX",
        "oanda": "NAS100_USD",
        "tv_ticker": "TVC:IXIC",
        "flag": "💻",
        "desc": "Nasdaq 100 Index / CFD",
        "market": "cfd"
    },
    "Russell 2000 (RUT)": {
        "symbol": "^RUT",
        "oanda": "US2000_USD",
        "tv_ticker": "TVC:RUT",
        "flag": "🚀",
        "desc": "Russell 2000 Index / CFD",
        "market": "cfd"
    },
    "Nikkei 225 (NI225)": {
        "symbol": "^N225",
        "oanda": "JP225_USD",
        "tv_ticker": "TVC:NI225",
        "flag": "🇯🇵",
        "desc": "Nikkei 225 Index / CFD",
        "market": "cfd"
    },
    "10Y Treasury (TNX)": {
        "symbol": "^TNX",
        "oanda": "USB10Y_USD",
        "tv_ticker": "TVC:US10Y",
        "flag": "🏛️",
        "desc": "10-Year U.S. Treasury Yield Index / CFD",
        "market": "cfd"
    },
    "WTI Crude (WTI)": {
        "symbol": "CL=F",
        "oanda": "WTICO_USD",
        "tv_ticker": "TVC:USOIL",
        "flag": "🛢️",
        "desc": "WTI Light Sweet Crude Oil CFD",
        "market": "cfd"
    },
    "Brent Crude (BRENT)": {
        "symbol": "BZ=F",
        "oanda": "BCO_USD",
        "tv_ticker": "TVC:UKOIL",
        "flag": "🌊",
        "desc": "Brent Crude Oil CFD",
        "market": "cfd"
    }
}

def fetch_tradingview_scan(timeframe: str = "5m") -> dict:
    """Pulls live indicators directly from TradingView's official CFD & Global Scanner API."""
    cols = ["close", "change", "RSI", "VWAP", "Recommend.All", "volume", "ATR", "open", "high", "low"]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    tv_data = {}
    
    tickers = [meta["tv_ticker"] for meta in ASSETS.values()]
    payload = {"symbols": {"tickers": tickers}, "columns": cols}
    
    for scan_endpoint in ["global", "cfd"]:
        try:
            url_scan = f"https://scanner.tradingview.com/{scan_endpoint}/scan"
            res = requests.post(url_scan, json=payload, headers=headers, timeout=8).json()
            for item in res.get("data", []):
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
            print(f"Error fetching TradingView scan ({scan_endpoint}): {e}")
            
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

def fetch_cfd_candles(symbol: str, timeframe: str = "5m", oanda_inst: str = None):
    oanda_key = os.getenv("OANDA_API_KEY", "").strip()
    if oanda_key and oanda_inst:
        try:
            from oanda_feed import fetch_oanda_candles
            odf = fetch_oanda_candles(instrument=oanda_inst, timeframe=timeframe, count=100, api_key=oanda_key)
            records = []
            for t, row in odf.iterrows():
                records.append({
                    "Timestamp": int(t.timestamp()),
                    "Open": row["Open"],
                    "High": row["High"],
                    "Low": row["Low"],
                    "Close": row["Close"],
                    "Volume": row["Volume"]
                })
            return pd.DataFrame(records)
        except Exception as e:
            print(f"OANDA API fetch failed for {oanda_inst}: {e}")

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
                "instructions": "Determine tactical positioning for the next 10 minutes (10 bars forward on 1-minute interval) based on TradingView & OANDA CFD momentum, VWAP deviation, and technical ratings.",
                "criteria": {
                    "BUY_LONG": "Clear bullish momentum above VWAP with high probability of higher prices over next 10 minutes",
                    "HOLD_CASH": "Consolidation, neutral chop, or tight range near VWAP over next 10 minutes",
                    "SELL_SHORT": "Clear bearish breakdown below VWAP with high probability of lower prices over next 10 minutes"
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

def analyze_asset(df, name="S&P 500 (SPX)", timeframe="1m", tv_metric=None):
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
    ret_5 = (close_vals[-1] / close_vals[-6] - 1) * 100 if len(close_vals) > 6 else 0.0
    ret_10 = (close_vals[-1] / close_vals[-11] - 1) * 100 if len(close_vals) > 11 else 0.0
    
    tv_ticker = ASSETS[name]["tv_ticker"]
    if tv_metric and tv_metric.get("close") is not None:
        price = float(tv_metric["close"])
        session_change = float(tv_metric["change"]) if tv_metric["change"] is not None else (close_vals[-1] / open_p[0] - 1) * 100
        rsi_14 = float(tv_metric["rsi"]) if tv_metric["rsi"] is not None else 50.0
        vwap = float(tv_metric["vwap"]) if tv_metric["vwap"] is not None else float(np.mean(close_vals[-20:]))
        tv_rating_score = tv_metric.get("recommend")
        feed_source = "TradingView CFD Live"
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
        feed_source = "CFD Chart Feed"

    vwap_z = (price - vwap) / (atr_14 + 1e-9)
    tv_rating_label, tv_rating_color = get_tv_rating_text(tv_rating_score)

    state_str = (
        f"CFD Asset: {name} (TradingView Symbol: {tv_ticker}, 1-Minute Horizon). "
        f"Forecasting Objective: Next 10 Minutes Movement (10 bars forward). "
        f"TradingView CFD Price: ${price:,.2f}. "
        f"TradingView CFD VWAP: ${vwap:,.2f} (Distance: {vwap_z:+.2f} ATRs). "
        f"TradingView 14-RSI: {rsi_14:.1f}. "
        f"TradingView Technical Rating: {tv_rating_label} ({tv_rating_score if tv_rating_score is not None else 0:+.2f}). "
        f"Micro-Momentum: 3-min={ret_3:+.2f}%, 5-min={ret_5:+.2f}%, 10-min={ret_10:+.2f}%. Session Return: {session_change:+.2f}%."
    )
    
    try:
        jev_res = call_jev_api(state_str)
        jev_choice = jev_res["choice"]
        prob_up = float(jev_res["prob_up"])
        confidence = float(jev_res["confidence"])
        is_live_jev = True
    except Exception:
        raw_score = 0.30 * ret_3 + 0.35 * ret_5 + 0.35 * ret_10 + 0.25 * (vwap_z * 0.4)
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
    decimals = 3 if "^TNX" in ASSETS[name]["symbol"] else 2
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
    if "S&P 500 (SPX)" in results and "Nasdaq 100 (NDX)" in results and "Russell 2000 (RUT)" in results and "Nikkei 225 (NI225)" in results:
        sp = results["S&P 500 (SPX)"]
        ndx = results["Nasdaq 100 (NDX)"]
        rut = results["Russell 2000 (RUT)"]
        nik = results["Nikkei 225 (NI225)"]
        
        tech_spread = ndx["ret_6"] - sp["ret_6"]
        small_spread = rut["ret_6"] - sp["ret_6"]
        
        if tech_spread > 0.15:
            insights.append({"type": "bull", "text": "Tech CFD Leadership (NDX > SPX): Nasdaq 100 CFD outperforming broader index CFDs."})
        elif tech_spread < -0.15:
            insights.append({"type": "warn", "text": "Tech CFD Drag (NDX < SPX): Tech CFD lagging broader market momentum."})
            
        if small_spread > 0.20:
            insights.append({"type": "bull", "text": "Broad Risk-On Breadth (RUT > SPX): Small-cap Russell CFD showing aggressive beta leadership."})
        elif small_spread < -0.20:
            insights.append({"type": "warn", "text": "Defensive Posture (RUT < SPX): Russell CFD lagging; watch for false large-cap breakouts."})
            
        if nik["prob_up"] > 0.55 or nik.get("tv_rating") in ["BUY", "STRONG BUY"]:
            insights.append({"type": "info", "text": f"Nikkei 225 CFD Signal: Asian session {nik.get('tv_rating', 'BULLISH')} momentum bias."})

    # Rates & Macro Insights with 10Y Treasury CFD
    if "10Y Treasury (TNX)" in results and "S&P 500 (SPX)" in results:
        tnx = results["10Y Treasury (TNX)"]
        sp = results["S&P 500 (SPX)"]
        if tnx["ret_6"] > 0.10 and sp["ret_6"] < -0.10:
            insights.append({"type": "info", "text": "🏛️ Rate Surge & Risk Off: 10Y Treasury Yield CFD rising while equity CFDs pull back."})
        elif tnx["ret_6"] < -0.10 and sp["ret_6"] > 0.10:
            insights.append({"type": "bull", "text": "⚡ Growth Reflation: 10Y Yield CFD softening as equity CFDs accelerate higher."})
        elif tnx.get("tv_rating") in ["BUY", "STRONG BUY"]:
            insights.append({"type": "warn", "text": "⚠️ Yield Pressure (TNX Rising): 10Y Yields pushing higher; watch duration headwinds for Nasdaq 100 CFD (NDX)."})

    # Energy Insights with WTI & Brent Crude CFDs
    if "WTI Crude (WTI)" in results and "Brent Crude (BRENT)" in results:
        wti = results["WTI Crude (WTI)"]
        brent = results["Brent Crude (BRENT)"]
        spread = brent["price"] - wti["price"]
        if brent["ret_6"] > 0.20 and wti["ret_6"] > 0.20:
            insights.append({"type": "warn", "text": f"🛢️ Crude Energy Surge: WTI (${wti['price']:.2f}) & Brent (${brent['price']:.2f}) accelerating higher. Brent-WTI Spread: ${spread:.2f}."})
        elif brent["ret_6"] < -0.20 and wti["ret_6"] < -0.20:
            insights.append({"type": "bull", "text": f"🌊 Deflationary Energy Relief: Crude oil pulling back (WTI {wti['ret_6']:+.2f}%, Brent {brent['ret_6']:+.2f}%), easing macro pressure."})

    return insights

@app.get("/")
def serve_home():
    html_path = Path(__file__).parent.parent / "public" / "index.html"
    if not html_path.exists():
        html_path = Path("public/index.html")
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Global Multi-CFD Momentum Radar is Running</h1>")

@app.get("/api/radar")
@app.get("/radar")
def get_radar(timeframe: str = "1m"):
    if timeframe not in ["1m", "5m", "1h", "1d"]:
        timeframe = "1m"

    tv_scan_tf = "5m" if timeframe == "1m" else timeframe
    tv_data = fetch_tradingview_scan(tv_scan_tf)

    results = {}
    for name, meta in ASSETS.items():
        try:
            df = fetch_cfd_candles(meta["symbol"], timeframe, oanda_inst=meta.get("oanda"))
            tv_metric = tv_data.get(meta["tv_ticker"])
            sig = analyze_asset(df, name=name, timeframe=timeframe, tv_metric=tv_metric)
            if sig:
                results[name] = sig
        except Exception as e:
            print(f"Error fetching {name}: {e}")

    insights = generate_insights(results)
    return {
        "status": "success",
        "feed": "TradingView Official CFD Feeds",
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
