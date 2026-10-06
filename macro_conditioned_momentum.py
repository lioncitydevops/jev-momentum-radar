"""
Macro-Conditioned Multi-Horizon Trend Signal Models for Equity Indices
(S&P 500, Nasdaq 100, Russell 2000, Nikkei 225)

Model A: Full Macro Cross-Asset Model (Affected by 10Y Treasury Yield + Brent Crude + WTI Crude)
Model B: Rate-Only Macro Model (Affected by 10Y Treasury Yield ONLY, excluding Brent & WTI)

Author: Quantitative Trader & Mathematician
Engine: TypeSafe Jev System One (jev-latest) / Calibrated Cross-Asset Prior
"""

import os
import json
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from multi_horizon_momentum import compute_1m_features, compute_indicative_prices

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

def format_macro_full_prompt(equity_name: str, eq_feat: dict, tnx_feat: dict, brent_feat: dict, wti_feat: dict, cs_context: dict = None) -> str:
    """
    Model A State Prompt: Equity Index state conditioned on 10Y Treasury Yield (TNX),
    Brent Crude, WTI Crude signals, Cross-Sectional Spreads & Rankings, and nanoHUB Wave Mechanics.
    """
    cs_block = ""
    if cs_context:
        rank_str = cs_context.get("equity_ranks", {}).get(equity_name, "N/A")
        cs_block = (
            f"--- Cross-Sectional Relative Strength & Spreads ---\n"
            f"Asset Relative Momentum Rank: #{rank_str} of 4 Equities\n"
            f"Tech Spread (NDX - SPX): {cs_context.get('tech_spread', 0.0):+.2f}%\n"
            f"Beta Spread (RUT - SPX): {cs_context.get('beta_spread', 0.0):+.2f}%\n"
            f"Asia-US Spread (NI225 - SPX): {cs_context.get('asia_spread', 0.0):+.2f}%\n"
            f"Yield Spillover Spread (TNX - SPX): {cs_context.get('rate_equity_spread', 0.0):+.2f}%\n"
            f"Energy Spillover Spread (Crude - SPX): {cs_context.get('energy_equity_spread', 0.0):+.2f}%\n"
        )

    w_dyn = eq_feat.get("wave_dynamics", {})
    w_block = ""
    if w_dyn:
        w_block = (
            f"--- nanoHUB Wave Physics & Continuous Propagator (Datta Framework) ---\n"
            f"Eigenvalues: λ = -γ ± iω_d (γ={w_dyn.get('gamma',0):.3f}, ω_0={w_dyn.get('omega_0',0):.3f}, Q-Factor={w_dyn.get('quality_factor_Q',0):.2f})\n"
            f"Matrix Exponential Propagated States (e^[A]τ): 1m_z={w_dyn.get('z_pred_1m',0):+.2f}σ, 10m_z={w_dyn.get('z_pred_10m',0):+.2f}σ, 30m_z={w_dyn.get('z_pred_30m',0):+.2f}σ, 1h_z={w_dyn.get('z_pred_1h',0):+.2f}σ\n"
            f"Multi-Horizon Normal Modes: Group Velocity v_g={w_dyn.get('group_velocity',0):+.4f}, Phase Shift Δϕ={w_dyn.get('wave_phase_shift_deg',0):.1f}°\n"
            f"Wave Interference Regime: {w_dyn.get('wave_interference_type','N/A')}\n"
        )

    return (
        f"Asset: {equity_name} (1-Minute Input Interval Bars)\n"
        f"Model Variant: MODEL A - FULL MACRO CROSS-ASSET (10Y Yield + Brent Crude + WTI Crude Conditioned)\n"
        f"Target Horizons: 1-Minute Forward (1 bar), 10-Minute Forward (10 bars), 30-Minute Forward (30 bars), 1-Hour Forward (60 bars)\n"
        f"--- Equity Microstructure ({equity_name}) ---\n"
        f"Latest Close Price: ${eq_feat['price']:,.2f}\n"
        f"VWAP Z-Score: {eq_feat['vwap_z']:+.2f} ATRs, RSI(14): {eq_feat['rsi_14']:.1f}, RVOL: {eq_feat['rvol']:.2f}x\n"
        f"Micro-Returns Curve: 1m={eq_feat['ret_1m']:+.2f}%, 3m={eq_feat['ret_3m']:+.2f}%, 5m={eq_feat['ret_5m']:+.2f}%, 10m={eq_feat['ret_10m']:+.2f}%, 30m={eq_feat['ret_30m']:+.2f}%, 60m={eq_feat.get('ret_60m', eq_feat['ret_30m']):+.2f}%\n"
        f"Temporal Slopes: Micro-vs-Mid={eq_feat.get('slope_micro_vs_mid', 0.0):+.3f}%, Mid-vs-Macro={eq_feat.get('slope_mid_vs_macro', 0.0):+.3f}%\n"
        f"Temporal Term Structure Regime: {eq_feat.get('term_structure_type', 'N/A')}\n"
        f"{cs_block}"
        f"{w_block}"
        f"--- Cross-Asset Macro Inputs ---\n"
        f"10Y Treasury Yield (TNX): ${tnx_feat['price']:.3f} | 5m Ret={tnx_feat['ret_5m']:+.2f}%, 10m Ret={tnx_feat['ret_10m']:+.2f}%, VWAP Z={tnx_feat['vwap_z']:+.2f}σ\n"
        f"Brent Crude Oil (BRENT): ${brent_feat['price']:.2f} | 5m Ret={brent_feat['ret_5m']:+.2f}%, 10m Ret={brent_feat['ret_10m']:+.2f}%, VWAP Z={brent_feat['vwap_z']:+.2f}σ\n"
        f"WTI Crude Oil (WTI): ${wti_feat['price']:.2f} | 5m Ret={wti_feat['ret_5m']:+.2f}%, 10m Ret={wti_feat['ret_10m']:+.2f}%, VWAP Z={wti_feat['vwap_z']:+.2f}σ\n"
        f"Macro Guidance: Rising yields (TNX) compress duration equity multiples. Surging crude oil (Brent/WTI) increases inflation drag and margin pressure."
    )

def format_rate_only_prompt(equity_name: str, eq_feat: dict, tnx_feat: dict, cs_context: dict = None) -> str:
    """
    Model B State Prompt: Equity Index state conditioned on 10Y Treasury Yield (TNX) ONLY,
    excluding Brent and WTI crude oil.
    """
    cs_block = ""
    if cs_context:
        rank_str = cs_context.get("equity_ranks", {}).get(equity_name, "N/A")
        cs_block = (
            f"--- Cross-Sectional Relative Strength & Spreads ---\n"
            f"Asset Relative Momentum Rank: #{rank_str} of 4 Equities\n"
            f"Tech Spread (NDX - SPX): {cs_context.get('tech_spread', 0.0):+.2f}%\n"
            f"Beta Spread (RUT - SPX): {cs_context.get('beta_spread', 0.0):+.2f}%\n"
            f"Yield Spillover Spread (TNX - SPX): {cs_context.get('rate_equity_spread', 0.0):+.2f}%\n"
        )

    w_dyn = eq_feat.get("wave_dynamics", {})
    w_block = ""
    if w_dyn:
        w_block = (
            f"--- nanoHUB Wave Physics & Continuous Propagator (Datta Framework) ---\n"
            f"Eigenvalues: λ = -γ ± iω_d (γ={w_dyn.get('gamma',0):.3f}, ω_0={w_dyn.get('omega_0',0):.3f}, Q-Factor={w_dyn.get('quality_factor_Q',0):.2f})\n"
            f"Matrix Exponential Propagated States (e^[A]τ): 1m_z={w_dyn.get('z_pred_1m',0):+.2f}σ, 10m_z={w_dyn.get('z_pred_10m',0):+.2f}σ, 30m_z={w_dyn.get('z_pred_30m',0):+.2f}σ, 1h_z={w_dyn.get('z_pred_1h',0):+.2f}σ\n"
            f"Multi-Horizon Normal Modes: Group Velocity v_g={w_dyn.get('group_velocity',0):+.4f}, Phase Shift Δϕ={w_dyn.get('wave_phase_shift_deg',0):.1f}°\n"
            f"Wave Interference Regime: {w_dyn.get('wave_interference_type','N/A')}\n"
        )

    return (
        f"Asset: {equity_name} (1-Minute Input Interval Bars)\n"
        f"Model Variant: MODEL B - RATE-ONLY MACRO (10Y Yield Conditioned, Excluding Crude Oil)\n"
        f"Target Horizons: 1-Minute Forward (1 bar), 10-Minute Forward (10 bars), 30-Minute Forward (30 bars), 1-Hour Forward (60 bars)\n"
        f"--- Equity Microstructure ({equity_name}) ---\n"
        f"Latest Close Price: ${eq_feat['price']:,.2f}\n"
        f"VWAP Z-Score: {eq_feat['vwap_z']:+.2f} ATRs, RSI(14): {eq_feat['rsi_14']:.1f}, RVOL: {eq_feat['rvol']:.2f}x\n"
        f"Micro-Returns Curve: 1m={eq_feat['ret_1m']:+.2f}%, 3m={eq_feat['ret_3m']:+.2f}%, 5m={eq_feat['ret_5m']:+.2f}%, 10m={eq_feat['ret_10m']:+.2f}%, 30m={eq_feat['ret_30m']:+.2f}%, 60m={eq_feat.get('ret_60m', eq_feat['ret_30m']):+.2f}%\n"
        f"Temporal Slopes: Micro-vs-Mid={eq_feat.get('slope_micro_vs_mid', 0.0):+.3f}%, Mid-vs-Macro={eq_feat.get('slope_mid_vs_macro', 0.0):+.3f}%\n"
        f"Temporal Term Structure Regime: {eq_feat.get('term_structure_type', 'N/A')}\n"
        f"{cs_block}"
        f"{w_block}"
        f"--- Rate Input (TNX Only) ---\n"
        f"10Y Treasury Yield (TNX): ${tnx_feat['price']:.3f} | 5m Ret={tnx_feat['ret_5m']:+.2f}%, 10m Ret={tnx_feat['ret_10m']:+.2f}%, VWAP Z={tnx_feat['vwap_z']:+.2f}σ\n"
        f"Macro Guidance: Evaluate equity forward trend strictly against interest rate duration sensitivity (10Y yield direction and volatility)."
    )

def simulate_macro_full_prior(equity_name: str, eq_feat: dict, tnx_feat: dict, brent_feat: dict, wti_feat: dict) -> dict:
    """
    Model A Calibrated Fallback Prior:
    Adjusts equity base score using 10Y Yield sensitivity + Brent/WTI Crude sensitivity + nanoHUB Wave Matrix Exp predictions.
    """
    ret_1m = eq_feat["ret_1m"]
    ret_3m = eq_feat["ret_3m"]
    ret_5m = eq_feat["ret_5m"]
    ret_10m = eq_feat["ret_10m"]
    ret_30m = eq_feat["ret_30m"]
    ret_60m = eq_feat.get("ret_60m", ret_30m)
    vwap_z = eq_feat["vwap_z"]

    w_dyn = eq_feat.get("wave_dynamics", {})
    z_w_1m = w_dyn.get("z_pred_1m", 0.0)
    z_w_10m = w_dyn.get("z_pred_10m", 0.0)
    z_w_30m = w_dyn.get("z_pred_30m", 0.0)
    z_w_1h = w_dyn.get("z_pred_1h", 0.0)

    # Base equity score incorporating wave exponential projections
    base_score = 0.20 * ret_1m + 0.25 * ret_3m + 0.25 * ret_5m + 0.20 * ret_10m + 0.25 * (vwap_z * 0.4) + 0.25 * (z_w_10m * 0.3)

    # 10Y Yield Sensitivity Factor
    rate_sens = 0.45 if "NDX" in equity_name else (0.40 if "RUT" in equity_name else 0.30)
    tnx_impact = - rate_sens * tnx_feat["ret_5m"] - 0.20 * tnx_feat["vwap_z"]

    # Energy Inflation Sensitivity Factor
    energy_ret = (brent_feat["ret_5m"] + wti_feat["ret_5m"]) / 2.0
    energy_z = (brent_feat["vwap_z"] + wti_feat["vwap_z"]) / 2.0
    energy_sens = 0.25 if "RUT" in equity_name else 0.20
    energy_impact = - energy_sens * energy_ret - 0.10 * energy_z

    adjusted_score = base_score + tnx_impact + energy_impact

    # Probabilities for 1m, 10m, 30m, 1h incorporating specific wave horizon projections
    prob_1m = float(np.clip(1.0 / (1.0 + np.exp(-(0.03 + 0.50 * (0.4 * ret_1m + 0.3 * (z_w_1m * 0.3) + 0.1 * tnx_impact)))), 0.05, 0.95))
    prob_10m = float(np.clip(1.0 / (1.0 + np.exp(-(0.08 + 0.40 * (adjusted_score + 0.15 * z_w_10m + 0.05 * (tnx_impact + energy_impact))))), 0.05, 0.95))
    prob_30m = float(np.clip(1.0 / (1.0 + np.exp(-(0.09 + 0.38 * (0.25 * ret_10m + 0.35 * ret_30m + 0.15 * z_w_30m + 0.10 * (tnx_impact + energy_impact))))), 0.05, 0.95))
    prob_1h = float(np.clip(1.0 / (1.0 + np.exp(-(0.10 + 0.35 * (0.20 * ret_30m + 0.35 * ret_60m + 0.15 * z_w_1h + 0.15 * (tnx_impact + energy_impact))))), 0.05, 0.95))

    def get_act(p, z):
        if p >= 0.60 and z > 0.10: return "STRONG_LONG"
        elif p >= 0.53: return "LEAN_LONG"
        elif p <= 0.40 and z < -0.10: return "STRONG_SHORT"
        elif p <= 0.47: return "LEAN_SHORT"
        return "NEUTRAL"

    probs = [prob_1m, prob_10m, prob_30m, prob_1h]
    avg_p = float(np.mean(probs))
    bull_c = sum(1 for p in probs if p > 0.53)
    bear_c = sum(1 for p in probs if p < 0.47)
    alignment = "STRONG_BULLISH_ALIGNMENT (4/4 Up)" if bull_c == 4 else ("STRONG_BEARISH_ALIGNMENT (4/4 Down)" if bear_c == 4 else ("MODERATE_BULLISH_BIAS (3/4 Up)" if bull_c >= 3 else ("MODERATE_BEARISH_BIAS (3/4 Down)" if bear_c >= 3 else "DIVERGENT_CHOP / NEUTRAL")))

    ip_data = compute_indicative_prices(eq_feat, prob_1m, prob_10m, prob_30m, prob_1h)

    return {
        "model_type": "MODEL_A_MACRO_FULL",
        "is_live_jev": False,
        "status": "simulation_mode",
        "alignment": alignment,
        "average_prob_up": round(avg_p, 4),
        "forward_projections": ip_data["forward_projections"],
        "macro_impacts": {
            "rate_impact_tnx": round(tnx_impact, 4),
            "energy_impact_crude": round(energy_impact, 4)
        },
        "forward_1m": {"horizon": "1min forward", "action": get_act(prob_1m, vwap_z), "prob_up": round(prob_1m, 4), "confidence": 0.72, **ip_data["h1m"]},
        "forward_10m": {"horizon": "10min forward", "action": get_act(prob_10m, vwap_z), "prob_up": round(prob_10m, 4), "confidence": 0.76, **ip_data["h10m"]},
        "forward_30m": {"horizon": "30min forward", "action": get_act(prob_30m, vwap_z), "prob_up": round(prob_30m, 4), "confidence": 0.78, **ip_data["h30m"]},
        "forward_1h": {"horizon": "1h forward", "action": get_act(prob_1h, vwap_z), "prob_up": round(prob_1h, 4), "confidence": 0.80, **ip_data["h1h"]}
    }

def simulate_rate_only_prior(equity_name: str, eq_feat: dict, tnx_feat: dict) -> dict:
    """
    Model B Calibrated Fallback Prior:
    Adjusts equity base score using 10Y Yield sensitivity ONLY (excluding Brent & WTI).
    """
    ret_1m = eq_feat["ret_1m"]
    ret_3m = eq_feat["ret_3m"]
    ret_5m = eq_feat["ret_5m"]
    ret_10m = eq_feat["ret_10m"]
    ret_30m = eq_feat["ret_30m"]
    ret_60m = eq_feat.get("ret_60m", ret_30m)
    vwap_z = eq_feat["vwap_z"]

    w_dyn = eq_feat.get("wave_dynamics", {})
    z_w_1m = w_dyn.get("z_pred_1m", 0.0)
    z_w_10m = w_dyn.get("z_pred_10m", 0.0)
    z_w_30m = w_dyn.get("z_pred_30m", 0.0)
    z_w_1h = w_dyn.get("z_pred_1h", 0.0)

    # Base equity score incorporating wave exponential projections
    base_score = 0.20 * ret_1m + 0.25 * ret_3m + 0.25 * ret_5m + 0.20 * ret_10m + 0.25 * (vwap_z * 0.4) + 0.25 * (z_w_10m * 0.3)

    # 10Y Yield Sensitivity Factor ONLY
    rate_sens = 0.50 if "NDX" in equity_name else (0.45 if "RUT" in equity_name else 0.35)
    tnx_impact = - rate_sens * tnx_feat["ret_5m"] - 0.25 * tnx_feat["vwap_z"]

    adjusted_score = base_score + tnx_impact

    # Probabilities for 1m, 10m, 30m, 1h incorporating specific wave horizon projections
    prob_1m = float(np.clip(1.0 / (1.0 + np.exp(-(0.03 + 0.50 * (0.4 * ret_1m + 0.3 * (z_w_1m * 0.3) + 0.1 * tnx_impact)))), 0.05, 0.95))
    prob_10m = float(np.clip(1.0 / (1.0 + np.exp(-(0.08 + 0.40 * (adjusted_score + 0.15 * z_w_10m)))), 0.05, 0.95))
    prob_30m = float(np.clip(1.0 / (1.0 + np.exp(-(0.09 + 0.38 * (0.25 * ret_10m + 0.35 * ret_30m + 0.15 * z_w_30m + 0.10 * tnx_impact)))), 0.05, 0.95))
    prob_1h = float(np.clip(1.0 / (1.0 + np.exp(-(0.10 + 0.35 * (0.20 * ret_30m + 0.35 * ret_60m + 0.15 * z_w_1h + 0.15 * tnx_impact)))), 0.05, 0.95))

    def get_act(p, z):
        if p >= 0.60 and z > 0.10: return "STRONG_LONG"
        elif p >= 0.53: return "LEAN_LONG"
        elif p <= 0.40 and z < -0.10: return "STRONG_SHORT"
        elif p <= 0.47: return "LEAN_SHORT"
        return "NEUTRAL"

    probs = [prob_1m, prob_10m, prob_30m, prob_1h]
    avg_p = float(np.mean(probs))
    bull_c = sum(1 for p in probs if p > 0.53)
    bear_c = sum(1 for p in probs if p < 0.47)
    alignment = "STRONG_BULLISH_ALIGNMENT (4/4 Up)" if bull_c == 4 else ("STRONG_BEARISH_ALIGNMENT (4/4 Down)" if bear_c == 4 else ("MODERATE_BULLISH_BIAS (3/4 Up)" if bull_c >= 3 else ("MODERATE_BEARISH_BIAS (3/4 Down)" if bear_c >= 3 else "DIVERGENT_CHOP / NEUTRAL")))

    ip_data = compute_indicative_prices(eq_feat, prob_1m, prob_10m, prob_30m, prob_1h)

    return {
        "model_type": "MODEL_B_RATE_ONLY",
        "is_live_jev": False,
        "status": "simulation_mode",
        "alignment": alignment,
        "average_prob_up": round(avg_p, 4),
        "forward_projections": ip_data["forward_projections"],
        "macro_impacts": {
            "rate_impact_tnx": round(tnx_impact, 4),
            "energy_impact_crude": 0.0 # Excluded by design in Model B
        },
        "forward_1m": {"horizon": "1min forward", "action": get_act(prob_1m, vwap_z), "prob_up": round(prob_1m, 4), "confidence": 0.72, **ip_data["h1m"]},
        "forward_10m": {"horizon": "10min forward", "action": get_act(prob_10m, vwap_z), "prob_up": round(prob_10m, 4), "confidence": 0.76, **ip_data["h10m"]},
        "forward_30m": {"horizon": "30min forward", "action": get_act(prob_30m, vwap_z), "prob_up": round(prob_30m, 4), "confidence": 0.78, **ip_data["h30m"]},
        "forward_1h": {"horizon": "1h forward", "action": get_act(prob_1h, vwap_z), "prob_up": round(prob_1h, 4), "confidence": 0.80, **ip_data["h1h"]}
    }

def query_typesafe_jev_macro(state_prompt: str, model_type: str, fallback_func, eq_feat: dict = None) -> dict:
    """Queries TypeSafe Jev System One API with macro-conditioned state representation."""
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        return fallback_func()

    payload = {
        "model": "jev-latest",
        "state": state_prompt,
        "questions": {
            "signal_1m_action": {
                "type": "choice",
                "instructions": f"Determine tactical directional signal for 1-MIN FORWARD horizon under {model_type}.",
                "criteria": {
                    "STRONG_LONG": "High probability of strong bullish upside continuation over 1-min forward horizon",
                    "LEAN_LONG": "Moderate bullish upside bias over 1-min forward horizon",
                    "NEUTRAL": "Mean-reverting consolidation, chop, or equilibrium over 1-min forward horizon",
                    "LEAN_SHORT": "Moderate bearish downside bias over 1-min forward horizon",
                    "STRONG_SHORT": "High probability of strong bearish downside continuation over 1-min forward horizon"
                }
            },
            "prob_up_1m": {
                "type": "noul",
                "instructions": "What is the continuous probability that price will close higher 1 minute from now?"
            },
            "signal_10m_action": {
                "type": "choice",
                "instructions": f"Determine tactical directional signal for 10-MIN FORWARD horizon under {model_type}.",
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
                "instructions": "What is the continuous probability that price will close higher 10 minutes from now?"
            },
            "signal_30m_action": {
                "type": "choice",
                "instructions": f"Determine tactical directional signal for 30-MIN FORWARD horizon under {model_type}.",
                "criteria": {
                    "STRONG_LONG": "High probability of strong bullish upside continuation over 30-min forward horizon",
                    "LEAN_LONG": "Moderate bullish upside bias over 30-min forward horizon",
                    "NEUTRAL": "Mean-reverting consolidation, chop, or equilibrium over 30-min forward horizon",
                    "LEAN_SHORT": "Moderate bearish downside bias over 30-min forward horizon",
                    "STRONG_SHORT": "High probability of strong bearish downside continuation over 30-min forward horizon"
                }
            },
            "prob_up_30m": {
                "type": "noul",
                "instructions": "What is the continuous probability that price will close higher 30 minutes from now?"
            },
            "signal_1h_action": {
                "type": "choice",
                "instructions": f"Determine tactical directional signal for 1-HOUR FORWARD horizon under {model_type}.",
                "criteria": {
                    "STRONG_LONG": "High probability of strong bullish upside continuation over 1-hour forward horizon",
                    "LEAN_LONG": "Moderate bullish upside bias over 1-hour forward horizon",
                    "NEUTRAL": "Mean-reverting consolidation, chop, or equilibrium over 1-hour forward horizon",
                    "LEAN_SHORT": "Moderate bearish downside bias over 1-hour forward horizon",
                    "STRONG_SHORT": "High probability of strong bearish downside continuation over 1-hour forward horizon"
                }
            },
            "prob_up_1h": {
                "type": "noul",
                "instructions": "What is the continuous probability that price will close higher 1 hour from now?"
            },
            "cross_sectional_leadership": {
                "type": "choice",
                "instructions": f"Evaluate this asset's relative strength and inter-market correlation position under {model_type}.",
                "criteria": {
                    "TOP_TIER_OUTPERFORMER": "Strongest relative momentum rank with macro yield/energy resilience",
                    "MID_PACK_PARTICIPANT": "In-line performance matching broader market index consensus",
                    "LAGGING_UNDERPERFORMER": "Lagging relative strength with acute macro yield or energy headwinds",
                    "FLIGHT_TO_SAFETY_HEDGE": "Counter-cyclical decorrelation during market stress or rate shock"
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

        act_1m = answers.get("signal_1m_action", {}).get("choice", "NEUTRAL")
        conf_1m = answers.get("signal_1m_action", {}).get("confidence", 0.70)
        p_1m = answers.get("prob_up_1m", {}).get("noul", 0.50)

        act_10m = answers.get("signal_10m_action", {}).get("choice", "NEUTRAL")
        conf_10m = answers.get("signal_10m_action", {}).get("confidence", 0.72)
        p_10m = answers.get("prob_up_10m", {}).get("noul", 0.50)

        act_30m = answers.get("signal_30m_action", {}).get("choice", "NEUTRAL")
        conf_30m = answers.get("signal_30m_action", {}).get("confidence", 0.75)
        p_30m = answers.get("prob_up_30m", {}).get("noul", 0.50)

        act_1h = answers.get("signal_1h_action", {}).get("choice", "NEUTRAL")
        conf_1h = answers.get("signal_1h_action", {}).get("confidence", 0.78)
        p_1h = answers.get("prob_up_1h", {}).get("noul", 0.50)

        probs = [p_1m, p_10m, p_30m, p_1h]
        avg_p = float(np.mean(probs))
        bull_c = sum(1 for p in probs if p > 0.53)
        bear_c = sum(1 for p in probs if p < 0.47)
        alignment = "STRONG_BULLISH_ALIGNMENT (4/4 Up)" if bull_c == 4 else ("STRONG_BEARISH_ALIGNMENT (4/4 Down)" if bear_c == 4 else ("MODERATE_BULLISH_BIAS (3/4 Up)" if bull_c >= 3 else ("MODERATE_BEARISH_BIAS (3/4 Down)" if bear_c >= 3 else "DIVERGENT_CHOP / NEUTRAL")))

        ip_extra = {}
        if eq_feat:
            ip_data = compute_indicative_prices(eq_feat, float(p_1m), float(p_10m), float(p_30m), float(p_1h))
            ip_extra["forward_projections"] = ip_data["forward_projections"]
            h1m_ext = ip_data["h1m"]
            h10m_ext = ip_data["h10m"]
            h30m_ext = ip_data["h30m"]
            h1h_ext = ip_data["h1h"]
        else:
            h1m_ext = h10m_ext = h30m_ext = h1h_ext = {}

        return {
            "model_type": model_type,
            "is_live_jev": True,
            "status": "live_jev_api",
            "alignment": alignment,
            "average_prob_up": round(avg_p, 4),
            **ip_extra,
            "forward_1m": {"horizon": "1min forward", "action": act_1m, "prob_up": round(float(p_1m), 4), "confidence": round(float(conf_1m), 4), **h1m_ext},
            "forward_10m": {"horizon": "10min forward", "action": act_10m, "prob_up": round(float(p_10m), 4), "confidence": round(float(conf_10m), 4), **h10m_ext},
            "forward_30m": {"horizon": "30min forward", "action": act_30m, "prob_up": round(float(p_30m), 4), "confidence": round(float(conf_30m), 4), **h30m_ext},
            "forward_1h": {"horizon": "1h forward", "action": act_1h, "prob_up": round(float(p_1h), 4), "confidence": round(float(conf_1h), 4), **h1h_ext},
            "raw_response": data
        }
    except Exception as e:
        print(f"[Warning] Jev API call failed ({e}). Using fallback prior.")
        return fallback_func()

def compute_cross_sectional_context(all_dfs: dict) -> dict:
    """
    Computes cross-sectional relative strength rankings and inter-market spreads
    across all available equity indices and macro assets.
    """
    if not all_dfs:
        return {}

    feats = {}
    for name, df in all_dfs.items():
        if df is not None and len(df) >= 15:
            feats[name] = compute_1m_features(df)

    equities = ["S&P 500 (SPX)", "Nasdaq 100 (NDX)", "Russell 2000 (RUT)", "Nikkei 225 (NI225)"]
    eq_rets = {}
    for eq in equities:
        if eq in feats:
            eq_rets[eq] = feats[eq]["ret_5m"]

    sorted_eqs = sorted(eq_rets.items(), key=lambda x: x[1], reverse=True)
    equity_ranks = {item[0]: i + 1 for i, item in enumerate(sorted_eqs)}

    spx_ret = feats.get("S&P 500 (SPX)", {}).get("ret_5m", 0.0)
    ndx_ret = feats.get("Nasdaq 100 (NDX)", {}).get("ret_5m", spx_ret)
    rut_ret = feats.get("Russell 2000 (RUT)", {}).get("ret_5m", spx_ret)
    ni_ret = feats.get("Nikkei 225 (NI225)", {}).get("ret_5m", spx_ret)

    tnx_ret = feats.get("10Y Treasury (TNX)", {}).get("ret_5m", 0.0)
    brent_ret = feats.get("Brent Crude (BRENT)", {}).get("ret_5m", 0.0)
    wti_ret = feats.get("WTI Crude (WTI)", {}).get("ret_5m", 0.0)

    tech_spread = float(np.round(ndx_ret - spx_ret, 3))
    beta_spread = float(np.round(rut_ret - spx_ret, 3))
    asia_spread = float(np.round(ni_ret - spx_ret, 3))

    rate_equity_spread = float(np.round(tnx_ret - spx_ret, 3))
    energy_equity_spread = float(np.round(((brent_ret + wti_ret) / 2.0) - spx_ret, 3))

    return {
        "equity_ranks": equity_ranks,
        "tech_spread": tech_spread,
        "beta_spread": beta_spread,
        "asia_spread": asia_spread,
        "rate_equity_spread": rate_equity_spread,
        "energy_equity_spread": energy_equity_spread
    }

def generate_macro_full_signals(eq_df: pd.DataFrame, tnx_df: pd.DataFrame, brent_df: pd.DataFrame, wti_df: pd.DataFrame, equity_name: str = "S&P 500 (SPX)", all_dfs: dict = None, loop_feedback: str = None) -> dict:
    """
    Model A Generator: Computes equity trend signals conditioned on 10Y Yield, Brent, WTI, and Cross-Sectional Spreads.
    """
    eq_feat = compute_1m_features(eq_df)
    tnx_feat = compute_1m_features(tnx_df)
    brent_feat = compute_1m_features(brent_df)
    wti_feat = compute_1m_features(wti_df)

    cs_context = compute_cross_sectional_context(all_dfs) if all_dfs else None
    prompt = format_macro_full_prompt(equity_name, eq_feat, tnx_feat, brent_feat, wti_feat, cs_context)
    if loop_feedback:
        prompt += f"\n--- Empirical Closed-Loop Feedback (Walk-Forward Autotuning) ---\n{loop_feedback}\n"

    fallback = lambda: simulate_macro_full_prior(equity_name, eq_feat, tnx_feat, brent_feat, wti_feat)
    signals = query_typesafe_jev_macro(prompt, "MODEL_A_MACRO_FULL", fallback)

    return {
        "model_variant": "MODEL_A_MACRO_FULL",
        "description": "Affected by 10Y Yield (TNX) + Brent Crude + WTI Crude + Cross-Sectional Spreads",
        "equity_name": equity_name,
        "cross_sectional_context": cs_context,
        "signals": signals,
        "state_prompt": prompt
    }

def generate_rate_only_signals(eq_df: pd.DataFrame, tnx_df: pd.DataFrame, equity_name: str = "S&P 500 (SPX)", all_dfs: dict = None, loop_feedback: str = None) -> dict:
    """
    Model B Generator: Computes equity trend signals conditioned on 10Y Yield ONLY + Cross-Sectional Spreads.
    """
    eq_feat = compute_1m_features(eq_df)
    tnx_feat = compute_1m_features(tnx_df)

    cs_context = compute_cross_sectional_context(all_dfs) if all_dfs else None
    prompt = format_rate_only_prompt(equity_name, eq_feat, tnx_feat, cs_context)
    if loop_feedback:
        prompt += f"\n--- Empirical Closed-Loop Feedback (Walk-Forward Autotuning) ---\n{loop_feedback}\n"

    fallback = lambda: simulate_rate_only_prior(equity_name, eq_feat, tnx_feat)
    signals = query_typesafe_jev_macro(prompt, "MODEL_B_RATE_ONLY", fallback)

    return {
        "model_variant": "MODEL_B_RATE_ONLY",
        "description": "Affected by 10Y Yield (TNX) ONLY + Cross-Sectional Spreads",
        "equity_name": equity_name,
        "cross_sectional_context": cs_context,
        "signals": signals,
        "state_prompt": prompt
    }

if __name__ == "__main__":
    print("=== Testing Macro-Conditioned Models A & B ===")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=40, freq="1min")
    
    make_df = func = lambda base, vol: pd.DataFrame({
        "Open": base + np.cumsum(np.random.normal(0, vol, 40)),
        "High": base + np.cumsum(np.random.normal(0, vol, 40)) + 0.2,
        "Low": base + np.cumsum(np.random.normal(0, vol, 40)) - 0.2,
        "Close": base + np.cumsum(np.random.normal(0, vol, 40)),
        "Volume": np.random.randint(200, 1000, 40)
    }, index=dates)

    sp_df = make_df(5800.0, 0.4)
    tnx_df = make_df(3.95, 0.005) # Yield surging
    brent_df = make_df(75.0, 0.1) # Crude surging
    wti_df = make_df(71.0, 0.1)

    print("\n--- Model A: Full Macro (10Y Yield + Brent + WTI) ---")
    res_a = generate_macro_full_signals(sp_df, tnx_df, brent_df, wti_df, "S&P 500 (SPX)")
    print(json.dumps(res_a["signals"], indent=2))

    print("\n--- Model B: Rate Only (10Y Yield Only) ---")
    res_b = generate_rate_only_signals(sp_df, tnx_df, "S&P 500 (SPX)")
    print(json.dumps(res_b["signals"], indent=2))
