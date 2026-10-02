"""
Multi-Horizon (5min, 10min, 15min) Forward Trend Signal Engine
Based on 1-Minute Input Interval Data

Author: Quantitative Trader & Mathematician
Engine: TypeSafe Jev System One (jev-latest) / Calibrated Mathematical Fallback
"""

import os
import json
import math
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

def compute_1m_features(df: pd.DataFrame) -> dict:
    """
    Computes mathematical features from 1-minute OHLCV candles:
    - Micro-momentum returns across lookbacks: 1m, 3m, 5m, 10m, 15m, 30m
    - Micro Garman-Klass realized volatility
    - Anchored Session VWAP and ATR-normalized distance (vwap_z)
    - 14-period Wilder RSI
    - Relative Volume (RVOL_20)
    - High/Low range position
    """
    data = df.copy()
    close = data["Close"].values
    high = data["High"].values
    low = data["Low"].values
    open_p = data["Open"].values
    volume = data["Volume"].fillna(0).values

    n = len(close)
    if n < 15:
        raise ValueError(f"Insufficient candle history ({n} bars). Need at least 15 1-minute bars.")

    # 1-minute Garman-Klass Volatility over last 15 bars
    log_hl = np.log(np.maximum(high[-15:], 1e-9) / np.maximum(low[-15:], 1e-9))
    log_co = np.log(np.maximum(close[-15:], 1e-9) / np.maximum(open_p[-15:], 1e-9))
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    daily_vol_est = float(np.sqrt(np.maximum(np.mean(gk_var), 1e-12))) # 1m bar vol

    # Returns over micro lookbacks (1m, 3m, 5m, 10m, 15m, 30m)
    ret_1m = float(np.round((close[-1] / close[-2] - 1) * 100, 3)) if n >= 2 else 0.0
    ret_3m = float(np.round((close[-1] / close[-4] - 1) * 100, 3)) if n >= 4 else 0.0
    ret_5m = float(np.round((close[-1] / close[-6] - 1) * 100, 3)) if n >= 6 else 0.0
    ret_10m = float(np.round((close[-1] / close[-11] - 1) * 100, 3)) if n >= 11 else 0.0
    ret_15m = float(np.round((close[-1] / close[-16] - 1) * 100, 3)) if n >= 16 else 0.0
    ret_30m = float(np.round((close[-1] / close[-31] - 1) * 100, 3)) if n >= 31 else ret_15m

    # Normalized Z-scores using Garman-Klass vol
    denom = (daily_vol_est + 1e-9)
    z_3m = float(np.round((ret_3m / 100.0) / (denom * np.sqrt(3)), 2))
    z_5m = float(np.round((ret_5m / 100.0) / (denom * np.sqrt(5)), 2))
    z_10m = float(np.round((ret_10m / 100.0) / (denom * np.sqrt(10)), 2))
    z_15m = float(np.round((ret_15m / 100.0) / (denom * np.sqrt(15)), 2))

    # Intraday ATR-14
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.std(close))
    atr_14 = max(atr_14, 0.01)

    # Session VWAP (or rolling window VWAP if session start is not distinct)
    total_vol = volume.sum()
    if total_vol > 0:
        typical_price = (high + low + close) / 3.0
        vwap = float((typical_price * volume).sum() / (total_vol + 1e-9))
    else:
        vwap = float(np.mean(close[-20:]))

    vwap_dist = close[-1] - vwap
    vwap_z = float(np.round(vwap_dist / atr_14, 2))

    # 14-period RSI
    delta = np.diff(close[-15:])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain)
    avg_loss = np.mean(loss)
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_14 = float(np.round(100 - (100 / (1 + rs)), 1))

    # Relative Volume (RVOL_20)
    recent_vol = volume[-1]
    avg_vol_20 = np.mean(volume[-20:]) if n >= 20 else np.mean(volume)
    rvol = float(np.round(recent_vol / (avg_vol_20 + 1e-9), 2))

    # Close location within recent 15-bar range (0.0 = Low, 1.0 = High)
    range_15 = max(high[-15:]) - min(low[-15:])
    close_loc = float(np.round((close[-1] - min(low[-15:])) / (range_15 + 1e-9), 2))

    # Momentum slope / acceleration (difference between 5m ret and 15m ret)
    mom_accel = float(np.round(ret_5m - (ret_15m / 3.0), 3))

    # Temporal Slope Analysis across horizons (micro 3m vs mid 10m, and mid 10m vs macro 30m)
    slope_micro_vs_mid = float(np.round(ret_3m - ret_10m, 3))
    slope_mid_vs_macro = float(np.round(ret_10m - ret_30m, 3))

    # Horizon Term Structure Classification
    if ret_3m > 0 and ret_5m > 0 and ret_10m > 0 and slope_micro_vs_mid > 0 and vwap_z > 0.1:
        term_structure_type = "BULLISH_ACCELERATING (Micro-momentum expanding above VWAP)"
    elif ret_3m < 0 and ret_5m < 0 and ret_10m < 0 and slope_micro_vs_mid < 0 and vwap_z < -0.1:
        term_structure_type = "BEARISH_BREAKDOWN (Micro-momentum deteriorating below VWAP)"
    elif rsi_14 > 70 and vwap_z > 1.8 and ret_1m < ret_3m:
        term_structure_type = "OVEREXTENDED_EXHAUSTION (Spiking stretch near Highs, vulnerable to 5m pullback)"
    elif rsi_14 < 30 and vwap_z < -1.8 and ret_1m > ret_3m:
        term_structure_type = "OVERCOLD_REBOUND (Extended breakdown near Lows, potential 5m mean-reversion bounce)"
    elif vwap_z > 0.1 and ret_3m < 0 and ret_10m > 0:
        term_structure_type = "PULLBACK_RECOVERY (Short-term 3m micro-dip within 10m/15m bullish trend)"
    else:
        term_structure_type = "NEUTRAL_CONSOLIDATION (Flat/divergent momentum across lookbacks near VWAP)"

    return {
        "price": float(close[-1]),
        "vwap": vwap,
        "vwap_z": vwap_z,
        "atr_14": atr_14,
        "rsi_14": rsi_14,
        "rvol": rvol,
        "close_loc": close_loc,
        "mom_accel": mom_accel,
        "slope_micro_vs_mid": slope_micro_vs_mid,
        "slope_mid_vs_macro": slope_mid_vs_macro,
        "term_structure_type": term_structure_type,
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "ret_5m": ret_5m,
        "ret_10m": ret_10m,
        "ret_15m": ret_15m,
        "ret_30m": ret_30m,
        "z_3m": z_3m,
        "z_5m": z_5m,
        "z_10m": z_10m,
        "z_15m": z_15m
    }

def format_multi_horizon_state_prompt(asset_name: str, features: dict, timestamp_str: str = None) -> str:
    """Formats 1-minute state representation prompt for TypeSafe Jev System One with rich temporal term structure awareness."""
    ts_text = f"Bar Timestamp: {timestamp_str}\n" if timestamp_str else ""
    return (
        f"Asset: {asset_name} (1-Minute Input Interval Bars)\n"
        f"{ts_text}"
        f"Target Horizons: 5-Minute Forward (5 bars), 10-Minute Forward (10 bars), 15-Minute Forward (15 bars)\n"
        f"--- Microstructure State ---\n"
        f"Latest 1m Close Price: ${features['price']:,.2f}\n"
        f"Anchored Session VWAP: ${features['vwap']:,.2f} (VWAP Distance Z-Score: {features['vwap_z']:+.2f} ATRs)\n"
        f"14-Period RSI: {features['rsi_14']:.1f} | RVOL: {features['rvol']:.2f}x | Range Location: {features['close_loc']:.2f}\n"
        f"--- Multi-Horizon Temporal Term Structure ---\n"
        f"Returns Curve: 1m={features['ret_1m']:+.2f}%, 3m={features['ret_3m']:+.2f}%, 5m={features['ret_5m']:+.2f}%, 10m={features['ret_10m']:+.2f}%, 15m={features['ret_15m']:+.2f}%, 30m={features['ret_30m']:+.2f}%\n"
        f"Normalized Z-Scores: Z(3m)={features['z_3m']:+.2f}σ, Z(5m)={features['z_5m']:+.2f}σ, Z(10m)={features['z_10m']:+.2f}σ, Z(15m)={features['z_15m']:+.2f}σ\n"
        f"Temporal Slopes: Micro vs Mid (3m-10m)={features['slope_micro_vs_mid']:+.3f}%, Mid vs Macro (10m-30m)={features['slope_mid_vs_macro']:+.3f}%\n"
        f"Momentum Acceleration: {features['mom_accel']:+.3f}%\n"
        f"Temporal Term Structure Regime: {features['term_structure_type']}"
    )

def simulate_calibrated_multi_horizon_prior(features: dict, tv_rating_score: float = 0.0) -> dict:
    """
    Calibrated quantitative logistic model for 5m, 10m, and 15m forward horizons
    used as fallback or offline simulation layer.
    """
    vwap_z = features["vwap_z"]
    ret_3m = features["ret_3m"]
    ret_5m = features["ret_5m"]
    ret_10m = features["ret_10m"]
    ret_15m = features["ret_15m"]
    rsi = features["rsi_14"]
    close_loc = features["close_loc"]

    # 1. 5-Minute Forward (Fast Micro-Burst, 5 bars forward)
    # Weights prioritize short 3m and 5m returns and micro VWAP stretch
    score_5m = 0.40 * ret_3m + 0.35 * ret_5m + 0.15 * ret_10m + 0.30 * (vwap_z * 0.4) + 0.15 * (close_loc - 0.5)
    if tv_rating_score:
        score_5m += 0.15 * tv_rating_score
    prob_5m = float(1.0 / (1.0 + np.exp(-(0.05 + 0.45 * score_5m))))
    prob_5m = float(np.clip(prob_5m, 0.05, 0.95))
    
    if prob_5m >= 0.60 and vwap_z > 0.10:
        action_5m = "STRONG_LONG"
    elif prob_5m >= 0.53:
        action_5m = "LEAN_LONG"
    elif prob_5m <= 0.40 and vwap_z < -0.10:
        action_5m = "STRONG_SHORT"
    elif prob_5m <= 0.47:
        action_5m = "LEAN_SHORT"
    else:
        action_5m = "NEUTRAL"

    # 2. 10-Minute Forward (Standard Intraday Cycle, 10 bars forward)
    # Weights balance 5m, 10m, and VWAP trend confirmation
    score_10m = 0.25 * ret_3m + 0.35 * ret_5m + 0.30 * ret_10m + 0.10 * ret_15m + 0.35 * (vwap_z * 0.4)
    if tv_rating_score:
        score_10m += 0.20 * tv_rating_score
    prob_10m = float(1.0 / (1.0 + np.exp(-(0.08 + 0.40 * score_10m))))
    prob_10m = float(np.clip(prob_10m, 0.05, 0.95))

    if prob_10m >= 0.60 and vwap_z > 0.15:
        action_10m = "STRONG_LONG"
    elif prob_10m >= 0.53:
        action_10m = "LEAN_LONG"
    elif prob_10m <= 0.40 and vwap_z < -0.15:
        action_10m = "STRONG_SHORT"
    elif prob_10m <= 0.47:
        action_10m = "LEAN_SHORT"
    else:
        action_10m = "NEUTRAL"

    # 3. 15-Minute Forward (Macro Intraday Continuation, 15 bars forward)
    # Weights emphasize 10m, 15m returns, VWAP regime, and RSI regime
    rsi_norm = (rsi - 50.0) / 25.0
    score_15m = 0.15 * ret_5m + 0.35 * ret_10m + 0.35 * ret_15m + 0.40 * (vwap_z * 0.4) + 0.10 * rsi_norm
    if tv_rating_score:
        score_15m += 0.25 * tv_rating_score
    prob_15m = float(1.0 / (1.0 + np.exp(-(0.10 + 0.35 * score_15m))))
    prob_15m = float(np.clip(prob_15m, 0.05, 0.95))

    if prob_15m >= 0.60 and vwap_z > 0.20:
        action_15m = "STRONG_LONG"
    elif prob_15m >= 0.53:
        action_15m = "LEAN_LONG"
    elif prob_15m <= 0.40 and vwap_z < -0.20:
        action_15m = "STRONG_SHORT"
    elif prob_15m <= 0.47:
        action_15m = "LEAN_SHORT"
    else:
        action_15m = "NEUTRAL"

    # Alignment evaluation across the 3 horizons
    probs = [prob_5m, prob_10m, prob_15m]
    avg_prob = float(np.mean(probs))
    bull_count = sum(1 for p in probs if p > 0.53)
    bear_count = sum(1 for p in probs if p < 0.47)

    if bull_count == 3:
        alignment = "STRONG_BULLISH_ALIGNMENT (3/3 Horizons Up)"
    elif bear_count == 3:
        alignment = "STRONG_BEARISH_ALIGNMENT (3/3 Horizons Down)"
    elif bull_count == 2:
        alignment = "MODERATE_BULLISH_BIAS (2/3 Horizons Up)"
    elif bear_count == 2:
        alignment = "MODERATE_BEARISH_BIAS (2/3 Horizons Down)"
    else:
        alignment = "DIVERGENT_CHOP / NEUTRAL"

    return {
        "is_live_jev": False,
        "status": "simulation_mode",
        "alignment": alignment,
        "average_prob_up": round(avg_prob, 4),
        "forward_5m": {
            "horizon": "5min forward (5 bars)",
            "action": action_5m,
            "prob_up": round(prob_5m, 4),
            "confidence": 0.75
        },
        "forward_10m": {
            "horizon": "10min forward (10 bars)",
            "action": action_10m,
            "prob_up": round(prob_10m, 4),
            "confidence": 0.78
        },
        "forward_15m": {
            "horizon": "15min forward (15 bars)",
            "action": action_15m,
            "prob_up": round(prob_15m, 4),
            "confidence": 0.80
        }
    }

def query_typesafe_jev_multi_horizon(state_prompt: str, features: dict, tv_rating_score: float = 0.0) -> dict:
    """
    Queries TypeSafe Jev System One API with 5m, 10m, and 15m forward trend signal questions.
    Falls back gracefully to mathematical simulation prior if key missing or request fails.
    """
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        return simulate_calibrated_multi_horizon_prior(features, tv_rating_score)

    payload = {
        "model": "jev-latest",
        "state": state_prompt,
        "questions": {
            "signal_5m_action": {
                "type": "choice",
                "instructions": "Determine tactical directional signal for the 5-MINUTE FORWARD horizon (5 bars ahead on 1-min input).",
                "criteria": {
                    "STRONG_LONG": "High probability of strong bullish upside continuation over 5-min forward horizon",
                    "LEAN_LONG": "Moderate bullish upside bias over 5-min forward horizon",
                    "NEUTRAL": "Mean-reverting consolidation, chop, or equilibrium over 5-min forward horizon",
                    "LEAN_SHORT": "Moderate bearish downside bias over 5-min forward horizon",
                    "STRONG_SHORT": "High probability of strong bearish downside continuation over 5-min forward horizon"
                }
            },
            "prob_up_5m": {
                "type": "noul",
                "instructions": "What is the continuous probability that price will close higher 5 minutes from now (next 5 bars on 1m data)?"
            },
            "signal_10m_action": {
                "type": "choice",
                "instructions": "Determine tactical directional signal for the 10-MINUTE FORWARD horizon (10 bars ahead on 1-min input).",
                "criteria": {
                    "STRONG_LONG": "High probability of strong bullish upside continuation over 10-min forward horizon",
                    "LEAN_LONG": "Moderate bullish upside bias over 10-min forward horizon",
                    "NEUTRAL": "Mean-reverting consolidation, chop, or equilibrium over 10-min forward horizon",
                    "LEAN_SHORT": "Moderate bearish downside bias over 10-min forward horizon",
                    "STRONG_SHORT": "High probability of strong bearish downside continuation over 10-min forward horizon"
                }
            },
            "prob_up_10m": {
                "type": "noul",
                "instructions": "What is the continuous probability that price will close higher 10 minutes from now (next 10 bars on 1m data)?"
            },
            "signal_15m_action": {
                "type": "choice",
                "instructions": "Determine tactical directional signal for the 15-MINUTE FORWARD horizon (15 bars ahead on 1-min input).",
                "criteria": {
                    "STRONG_LONG": "High probability of strong bullish upside continuation over 15-min forward horizon",
                    "LEAN_LONG": "Moderate bullish upside bias over 15-min forward horizon",
                    "NEUTRAL": "Mean-reverting consolidation, chop, or equilibrium over 15-min forward horizon",
                    "LEAN_SHORT": "Moderate bearish downside bias over 15-min forward horizon",
                    "STRONG_SHORT": "High probability of strong bearish downside continuation over 15-min forward horizon"
                }
            },
            "prob_up_15m": {
                "type": "noul",
                "instructions": "What is the continuous probability that price will close higher 15 minutes from now (next 15 bars on 1m data)?"
            },
            "horizon_term_structure": {
                "type": "choice",
                "instructions": "Evaluate the relationship across the 5m, 10m, and 15m forward horizons. Classify the multi-horizon temporal relationship.",
                "criteria": {
                    "UNIFORM_ACCELERATION": "Consistent momentum acceleration aligned across 5m, 10m, and 15m horizons",
                    "EXHAUSTION_PULLBACK": "Fast 5m micro-momentum is overstretched relative to 15m trend, signaling short-term pullback risk",
                    "PULLBACK_RECOVERY": "Short-term 5m pullback within a longer 15m trend, offering retest entry setup",
                    "MEAN_REVERTING_CHOP": "Divergent or flat momentum across horizons near equilibrium VWAP"
                }
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(TYPESAFE_URL, json=payload, headers=headers, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        answers = data.get("answers", {})

        act_5m = answers.get("signal_5m_action", {}).get("choice", "NEUTRAL")
        conf_5m = answers.get("signal_5m_action", {}).get("confidence", 0.70)
        p_5m = answers.get("prob_up_5m", {}).get("noul", 0.50)

        act_10m = answers.get("signal_10m_action", {}).get("choice", "NEUTRAL")
        conf_10m = answers.get("signal_10m_action", {}).get("confidence", 0.72)
        p_10m = answers.get("prob_up_10m", {}).get("noul", 0.50)

        act_15m = answers.get("signal_15m_action", {}).get("choice", "NEUTRAL")
        conf_15m = answers.get("signal_15m_action", {}).get("confidence", 0.75)
        p_15m = answers.get("prob_up_15m", {}).get("noul", 0.50)

        probs = [p_5m, p_10m, p_15m]
        avg_p = float(np.mean(probs))
        bull_c = sum(1 for p in probs if p > 0.53)
        bear_c = sum(1 for p in probs if p < 0.47)

        if bull_c == 3:
            alignment = "STRONG_BULLISH_ALIGNMENT (3/3 Horizons Up)"
        elif bear_c == 3:
            alignment = "STRONG_BEARISH_ALIGNMENT (3/3 Horizons Down)"
        elif bull_c == 2:
            alignment = "MODERATE_BULLISH_BIAS (2/3 Horizons Up)"
        elif bear_c == 2:
            alignment = "MODERATE_BEARISH_BIAS (2/3 Horizons Down)"
        else:
            alignment = "DIVERGENT_CHOP / NEUTRAL"

        return {
            "is_live_jev": True,
            "status": "live_jev_api",
            "alignment": alignment,
            "average_prob_up": round(avg_p, 4),
            "forward_5m": {
                "horizon": "5min forward (5 bars)",
                "action": act_5m,
                "prob_up": round(float(p_5m), 4),
                "confidence": round(float(conf_5m), 4)
            },
            "forward_10m": {
                "horizon": "10min forward (10 bars)",
                "action": act_10m,
                "prob_up": round(float(p_10m), 4),
                "confidence": round(float(conf_10m), 4)
            },
            "forward_15m": {
                "horizon": "15min forward (15 bars)",
                "action": act_15m,
                "prob_up": round(float(p_15m), 4),
                "confidence": round(float(conf_15m), 4)
            },
            "raw_response": data
        }
    except Exception as e:
        print(f"[Warning] Live TypeSafe Jev API call failed ({e}). Falling back to calibrated prior model.")
        return simulate_calibrated_multi_horizon_prior(features, tv_rating_score)

def generate_multi_horizon_signals(df_1m: pd.DataFrame, asset_name: str = "S&P 500 (SPX)", tv_rating_score: float = 0.0) -> dict:
    """
    Main entry point to compute 5-min forward, 10-min forward, and 15-min forward trend signals
    from 1-minute input candles.
    """
    features = compute_1m_features(df_1m)
    timestamp_str = df_1m.index[-1].strftime('%Y-%m-%d %H:%M:%S') if isinstance(df_1m.index, pd.DatetimeIndex) else None
    prompt = format_multi_horizon_state_prompt(asset_name, features, timestamp_str)
    jev_result = query_typesafe_jev_multi_horizon(prompt, features, tv_rating_score)

    return {
        "asset_name": asset_name,
        "input_interval": "1m",
        "timestamp": timestamp_str,
        "features": features,
        "state_prompt": prompt,
        "signals": jev_result
    }

if __name__ == "__main__":
    # Test script with synthetic or live 1-min data
    print("=== Testing Multi-Horizon (5m, 10m, 15m) Trend Signal Engine ===")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(tz="America/New_York"), periods=60, freq="1min")
    prices = 5700.0 + np.cumsum(np.random.normal(0.15, 0.4, len(dates)))
    synthetic_1m = pd.DataFrame({
        "Open": prices - np.random.uniform(0, 0.2, len(dates)),
        "High": prices + np.random.uniform(0.1, 0.5, len(dates)),
        "Low": prices - np.random.uniform(0.1, 0.5, len(dates)),
        "Close": prices,
        "Volume": np.random.randint(100, 1000, len(dates))
    }, index=dates)

    res = generate_multi_horizon_signals(synthetic_1m, "S&P 500 (SPX)")
    print(json.dumps(res, indent=2))
