"""
Maritime Tanker Traffic & Physical Supply-Conditioned Forward Estimation Model for Brent Crude
Integrates real-time maritime AIS shipping telemetry, chokepoint transit flows,
floating storage, crack spreads, and forward curve backwardation from Oil_Tanker_Traffic_AntiGravity.

Engine: TypeSafe Jev System One (jev-latest) + nanoHUB Damped Wave Propagator
Author: Quantitative Energy & Macro Strategist
"""

import os
import sys
import json
import math
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Setup logger
logger = logging.getLogger("maritime_brent")

# Add Oil_Tanker_Traffic_AntiGravity path dynamically if available
SHIPPING_APP_DIR = os.environ.get(
    "OIL_TANKER_APP_DIR",
    r"G:\My Drive\Oil_Tanker_Traffic_AntiGravity"
)
if os.path.exists(SHIPPING_APP_DIR) and SHIPPING_APP_DIR not in sys.path:
    sys.path.insert(0, SHIPPING_APP_DIR)

from multi_horizon_momentum import (
    compute_1m_features,
    compute_wave_oscillator_dynamics,
    _matrix_expm
)

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"


def get_live_shipping_telemetry() -> Dict[str, Any]:
    """
    Ingests live telemetry from Oil_Tanker_Traffic_AntiGravity engines.
    Provides graceful fallback to realistic calibrated shipping baselines if the path is unavailable.
    """
    telemetry = {
        "source": "fallback_model",
        "hormuz": {
            "active_vessels_counted": 2,
            "baseline_vessels": 46,
            "dark_crossings_estimated": 30,
            "clandestine_rate_pct": 93.8,
            "active_capacity_mbbls": 2.0,
            "status": "Severely Constricted / Dark Fleet Dominance"
        },
        "redsea_bab_el_mandeb": {
            "traffic_today": 12,
            "historical_5yr_avg": 42.0,
            "cape_diversion_delay_days": 12.0,
            "status": "Houthi Blockade / Rerouting via Cape of Good Hope"
        },
        "suez_canal": {
            "traffic_today": 18,
            "historical_5yr_avg": 24.0,
            "trend": "Diverted / Depressed"
        },
        "fujairah": {
            "tankers_anchored_today": 32,
            "seven_day_ma": 31.5,
            "turnover_outflow_offset_pct": 52.0,
            "floating_storage_mbbls": 18.2
        },
        "energy_state": {
            "brent_prompt_price": 104.50,
            "wti_prompt_price": 99.80,
            "tapis_asean_usd": 108.20,
            "singapore_gasoil_crack_usd": 28.50,
            "jet_fuel_crack_usd": 26.40,
            "gasoline_ron95_crack_usd": 18.20,
            "singapore_onshore_stocks_mbbls": 24.8,
            "fujairah_asean_floating_storage_mbbls": 18.2,
            "forward_curve_structure": "Inverted / Steep Backwardation",
            "prompt_to_m6_spread": "+$9.70/bbl",
            "m6_price": 94.80
        },
        "asian_demand_indices": {
            "china_import_index": 60.5,
            "china_reduction_pct": 39.5,
            "india_import_index": 138.2,
            "japan_index": 98.4,
            "skorea_index": 99.1
        },
        "forward_curve_strip": [
            {"month": "M0 (Prompt)", "price": 104.50},
            {"month": "M+1", "price": 102.80},
            {"month": "M+2", "price": 101.20},
            {"month": "M+3", "price": 98.60},
            {"month": "M+4", "price": 96.40},
            {"month": "M+5", "price": 94.80},
            {"month": "M+6", "price": 94.80},
            {"month": "M+12", "price": 87.50}
        ]
    }

    # Attempt dynamic import from Oil_Tanker_Traffic_AntiGravity
    try:
        from src.analytics.traffic_engine import traffic_engine
        from src.analytics.energy_engine import energy_engine
        from src.data.market_feed import get_market_intelligence

        t_metrics = traffic_engine.get_current_metrics()
        e_summary = energy_engine.get_summary()
        e_state = e_summary.get("state", {})
        mkt = get_market_intelligence()

        telemetry["source"] = "Oil_Tanker_Traffic_AntiGravity (Live Native)"
        if "hormuz" in t_metrics:
            telemetry["hormuz"] = {
                "active_vessels_counted": int(t_metrics["hormuz"].get("active_vessels_counted", 2)),
                "baseline_vessels": int(t_metrics["hormuz"].get("baseline_vessels", 46)),
                "dark_crossings_estimated": int(t_metrics["hormuz"].get("dark_crossings_estimated", 30)),
                "clandestine_rate_pct": float(t_metrics["hormuz"].get("clandestine_rate_pct", 93.8)),
                "active_capacity_mbbls": float(t_metrics["hormuz"].get("active_capacity_mbbls", 2.0)),
                "status": "Severely Constricted / Dark Fleet Dominance" if t_metrics["hormuz"].get("clandestine_rate_pct", 0) > 80 else "Normal Transits"
            }
        if "bab_el_mandeb" in t_metrics:
            telemetry["redsea_bab_el_mandeb"] = {
                "traffic_today": int(t_metrics["bab_el_mandeb"].get("traffic_today", 12)),
                "historical_5yr_avg": float(t_metrics["bab_el_mandeb"].get("historical_5yr_avg", 42.0)),
                "cape_diversion_delay_days": 12.0,
                "status": "Houthi Blockade / Tankers Diverted via Cape of Good Hope"
            }
        if "suez_canal" in t_metrics:
            telemetry["suez_canal"] = {
                "traffic_today": int(t_metrics["suez_canal"].get("traffic_today", 18)),
                "historical_5yr_avg": float(t_metrics["suez_canal"].get("historical_5yr_avg", 24.0)),
                "trend": t_metrics["suez_canal"].get("trend", "Diverted")
            }
        if "fujairah" in t_metrics:
            telemetry["fujairah"] = {
                "tankers_anchored_today": int(t_metrics["fujairah"].get("tankers_anchored_today", 32)),
                "seven_day_ma": float(t_metrics["fujairah"].get("seven_day_ma", 31.5)),
                "turnover_outflow_offset_pct": float(t_metrics["fujairah"].get("turnover_outflow_offset_pct", 52.0)),
                "floating_storage_mbbls": float(e_state.get("fujairah_asean_floating_storage_mbbls", 18.2))
            }
        if "asian_demand_indices" in t_metrics:
            adi = t_metrics["asian_demand_indices"]
            telemetry["asian_demand_indices"] = {
                "china_import_index": float(adi.get("china", 60.5)),
                "china_reduction_pct": float(adi.get("china_reduction_pct", 39.5)),
                "india_import_index": float(adi.get("india", 138.2)),
                "japan_index": float(adi.get("japan", 98.4)),
                "skorea_index": float(adi.get("skorea", 99.1))
            }

        # Energy & Forward curve state
        brent_prompt = float(mkt.get("brent_prompt_price", e_state.get("brent_prompt_usd", 104.50)))
        wti_prompt = float(mkt.get("wti_prompt_price", e_state.get("wti_crude_usd", 99.80)))
        dec_price = float(mkt.get("dec_horizon_price", 101.20))
        m6_price = round(brent_prompt - 9.70, 2)

        telemetry["energy_state"] = {
            "brent_prompt_price": brent_prompt,
            "wti_prompt_price": wti_prompt,
            "tapis_asean_usd": float(e_state.get("tapis_asean_usd", round(brent_prompt + 3.70, 2))),
            "singapore_gasoil_crack_usd": float(e_state.get("singapore_gasoil_crack_usd", 28.50)),
            "jet_fuel_crack_usd": float(e_state.get("jet_fuel_crack_usd", 26.40)),
            "gasoline_ron95_crack_usd": float(e_state.get("gasoline_ron95_crack_usd", 18.20)),
            "singapore_onshore_stocks_mbbls": float(e_state.get("singapore_onshore_stocks_mbbls", 24.8)),
            "fujairah_asean_floating_storage_mbbls": float(e_state.get("fujairah_asean_floating_storage_mbbls", 18.2)),
            "forward_curve_structure": mkt.get("forward_curve_structure", "Inverted / Steep Backwardation"),
            "prompt_to_m6_spread": f"+${round(brent_prompt - m6_price, 2):.2f}/bbl",
            "m6_price": m6_price
        }

        if "forward_curve" in mkt and isinstance(mkt["forward_curve"], list):
            telemetry["forward_curve_strip"] = mkt["forward_curve"]
    except Exception as e:
        logger.info(f"Using calibrated shipping telemetry: {e}")

    return telemetry


def compute_maritime_physical_supply_index(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes quantitative Maritime Physical Supply Index (MPSI) and composite Z-score.
    
    Components:
    1. Chokepoint Constriction (S_chokepoints): Hormuz dark fleet rate + Bab El-Mandeb diversion
    2. Refinery Demand Pull (S_refinery): Singapore Gasoil crack spread vs historical norm ($18/bbl)
    3. Storage Depletion (S_storage): Floating storage vs baseline buffer
    4. Forward Inversion Steepness (S_backwardation): Prompt vs M6 forward backwardation
    """
    h_dark_rate = telemetry.get("hormuz", {}).get("clandestine_rate_pct", 90.0)
    bem_today = telemetry.get("redsea_bab_el_mandeb", {}).get("traffic_today", 12)
    bem_avg = telemetry.get("redsea_bab_el_mandeb", {}).get("historical_5yr_avg", 42.0)
    cape_delay = telemetry.get("redsea_bab_el_mandeb", {}).get("cape_diversion_delay_days", 12.0)

    # 1. Chokepoints constriction score [-3.0 to +3.0]
    # High clandestine rate (>80%) and Bab El-Mandeb traffic deficit (>60% drop) + Cape delays = intense supply bottleneck
    bem_deficit = max(0.0, (bem_avg - bem_today) / max(1.0, bem_avg))
    s_chokepoints = float(np.clip((h_dark_rate / 35.0 - 1.5) + (bem_deficit * 2.5) + (cape_delay / 8.0 - 1.0), -3.0, 3.0))

    # 2. Refinery Margin Demand Pull [-3.0 to +3.0]
    # Singapore Gasoil crack > $25/bbl signals severe middle distillate squeeze pulling prompt barrels
    gasoil_crack = telemetry.get("energy_state", {}).get("singapore_gasoil_crack_usd", 28.5)
    s_refinery = float(np.clip((gasoil_crack - 20.0) / 4.0, -3.0, 3.0))

    # 3. Storage Buffer Score [-3.0 to +3.0]
    # Onshore & Floating stocks vs normal operating levels
    fl_storage = telemetry.get("energy_state", {}).get("fujairah_asean_floating_storage_mbbls", 18.2)
    s_storage = float(np.clip((25.0 - fl_storage) / 3.0, -3.0, 3.0))

    # 4. Backwardation Steepness Score [-3.0 to +3.0]
    prompt = telemetry.get("energy_state", {}).get("brent_prompt_price", 104.5)
    m6 = telemetry.get("energy_state", {}).get("m6_price", 94.8)
    backw_spread = prompt - m6
    s_backwardation = float(np.clip((backw_spread - 4.0) / 2.0, -3.0, 3.0))

    # Weighted Composite Z-Score (Bullish supply disruption > 0, Bearish oversupply < 0)
    z_maritime = float(
        0.35 * s_chokepoints +
        0.25 * s_refinery +
        0.20 * s_storage +
        0.20 * s_backwardation
    )
    z_maritime = float(np.round(np.clip(z_maritime, -3.0, 3.0), 3))

    # Regime categorization
    if z_maritime >= 1.5:
        regime = "ACUTE_PHYSICAL_DISRUPTION_BACKWARDATION"
        bias = "STRONG_BULLISH_SUPPLY_PIN"
    elif z_maritime >= 0.6:
        regime = "MODERATE_CHOKEPOINT_TIGHTNESS"
        bias = "BULLISH_FORWARD_PRESSURE"
    elif z_maritime <= -1.0:
        regime = "SUPPLY_SURFEIT_CONTANGO_RISK"
        bias = "BEARISH_DESTOCKING"
    else:
        regime = "EQUILIBRIUM_MARITIME_FLOW"
        bias = "NEUTRAL_FLOW"

    return {
        "z_maritime": z_maritime,
        "regime": regime,
        "bias": bias,
        "components": {
            "chokepoints_score": round(s_chokepoints, 2),
            "refinery_crack_score": round(s_refinery, 2),
            "storage_buffer_score": round(s_storage, 2),
            "backwardation_score": round(s_backwardation, 2),
        },
        "raw_metrics": {
            "hormuz_dark_pct": h_dark_rate,
            "cape_delay_days": cape_delay,
            "singapore_gasoil_crack": gasoil_crack,
            "prompt_to_m6_spread_usd": round(backw_spread, 2),
            "floating_storage_mbbls": fl_storage
        }
    }


def format_maritime_brent_prompt(
    brent_feat: dict,
    telemetry: dict,
    mpsi: dict,
    wti_feat: dict = None,
    loop_feedback: str = None
) -> str:
    """
    Constructs the TypeSafe Jev System One prompt fusing:
    1. Brent Microstructure & nanoHUB Wave Propagator
    2. Real-Time Shipping & AIS Chokepoint Telemetry
    3. Energy Refining Crack Spreads & Inventory
    4. Crude Futures Backwardation Forward Strip
    """
    w_dyn = brent_feat.get("wave_dynamics", {})
    w_block = ""
    if w_dyn:
        w_block = (
            f"--- nanoHUB Wave Physics Propagator (Datta Framework) ---\n"
            f"Oscillator Eigenvalues: λ = -γ ± iω_d (γ={w_dyn.get('gamma',0):.3f}, ω_0={w_dyn.get('omega_0',0):.3f}, Q-Factor={w_dyn.get('quality_factor_Q',0):.2f})\n"
            f"Continuous Propagated States (e^[A]τ): 1m_z={w_dyn.get('z_pred_1m',0):+.2f}σ, 10m_z={w_dyn.get('z_pred_10m',0):+.2f}σ, 30m_z={w_dyn.get('z_pred_30m',0):+.2f}σ, 1h_z={w_dyn.get('z_pred_1h',0):+.2f}σ\n"
            f"Wave Interference: Group Velocity v_g={w_dyn.get('group_velocity',0):+.4f}, Normal Mode={w_dyn.get('wave_interference_type','N/A')}\n"
        )

    wti_line = ""
    if wti_feat:
        spread = brent_feat['price'] - wti_feat['price']
        wti_line = f"WTI Crude Benchmark: ${wti_feat['price']:.2f} | 5m Ret={wti_feat.get('ret_5m',0.0):+.2f}% | Brent-WTI Spread: ${spread:.2f}/bbl\n"

    h = telemetry.get("hormuz", {})
    r = telemetry.get("redsea_bab_el_mandeb", {})
    f = telemetry.get("fujairah", {})
    e = telemetry.get("energy_state", {})
    ad = telemetry.get("asian_demand_indices", {})

    fb_block = f"--- Empirical Loop Feedback ---\n{loop_feedback}\n" if loop_feedback else ""

    return (
        f"Asset: Brent Crude Oil (BRENT / BCO_USD)\n"
        f"Model Variant: MARITIME TANKER TRAFFIC & PHYSICAL CHOKEPOINT FORWARD ESTIMATION\n"
        f"Target Forward Horizons: 1-Minute Forward (1 bar), 10-Minute Forward (10 bars), 30-Minute Forward (30 bars), 1-Hour Forward (60 bars)\n"
        f"--- Brent Crude Microstructure & Returns Curve ---\n"
        f"Prompt Spot/CFD Price: ${brent_feat['price']:.2f}\n"
        f"VWAP Z-Score: {brent_feat['vwap_z']:+.2f}σ, RSI(14): {brent_feat['rsi_14']:.1f}, RVOL: {brent_feat['rvol']:.2f}x\n"
        f"Micro-Returns Curve: 1m={brent_feat['ret_1m']:+.2f}%, 3m={brent_feat['ret_3m']:+.2f}%, 5m={brent_feat['ret_5m']:+.2f}%, 10m={brent_feat['ret_10m']:+.2f}%, 30m={brent_feat['ret_30m']:+.2f}%, 60m={brent_feat.get('ret_60m', brent_feat['ret_30m']):+.2f}%\n"
        f"{wti_line}"
        f"{w_block}"
        f"--- Real-Time Maritime AIS Tanker Traffic & Chokepoint Telemetry ---\n"
        f"Maritime Physical Supply Index (MPSI): Z={mpsi['z_maritime']:+.2f}σ ({mpsi['regime']})\n"
        f"Strait of Hormuz: {h.get('active_vessels_counted', 2)} active tankers (Baseline {h.get('baseline_vessels', 46)}). Clandestine Dark Fleet Rate: {h.get('clandestine_rate_pct', 93.8):.1f}% (Night transits via Fujairah STS).\n"
        f"Bab El-Mandeb / Red Sea: {r.get('traffic_today', 12)} tankers/day (5-Yr Avg: {r.get('historical_5yr_avg', 42.0)}). Cape of Good Hope Diversion Delay: +{r.get('cape_diversion_delay_days', 12.0):.0f} days (+35% ton-mile soak).\n"
        f"Fujairah Tanker Anchorage: {f.get('tankers_anchored_today', 32)} tankers, Turnover offset: {f.get('turnover_outflow_offset_pct', 52.0):.1f}%, Floating Storage: {f.get('floating_storage_mbbls', 18.2):.1f}M bbls.\n"
        f"Asian Seaborne Demand Divergence: China Import Index={ad.get('china_import_index', 60.5)} (-{ad.get('china_reduction_pct', 39.5):.1f}% vs baseline) vs India Import Index={ad.get('india_import_index', 138.2)} (+38.2%).\n"
        f"--- Energy Crack Spreads & Futures Curve Backwardation ---\n"
        f"Singapore Gasoil 10ppm Crack: ${e.get('singapore_gasoil_crack_usd', 28.50):.2f}/bbl (Refinery middle-distillate margin squeeze).\n"
        f"Jet Fuel A-1 Crack: ${e.get('jet_fuel_crack_usd', 26.40):.2f}/bbl | Tapis ASEAN Sweet Premium: ${e.get('tapis_asean_usd', 108.20):.2f}/bbl.\n"
        f"Forward Curve Structure: {e.get('forward_curve_structure', 'Inverted / Steep Backwardation')}\n"
        f"Prompt vs M+6 Forward Spread: {e.get('prompt_to_m6_spread', '+$9.70/bbl')} (Prompt ${e.get('brent_prompt_price', 104.50):.2f} vs M6 ${e.get('m6_price', 94.80):.2f}).\n"
        f"{fb_block}"
        f"Estimation Logic: Physical tanker bottlenecks and steep forward backwardation create prompt physical scarcity and upward forward skew. Short-term intraday pullbacks provide dip-buying absorption in persistent structural backwardation."
    )


def simulate_calibrated_maritime_brent_prior(
    brent_feat: dict,
    telemetry: dict,
    mpsi: dict,
    wti_feat: dict = None
) -> dict:
    """
    Mathematical Calibrated Fallback Prior:
    Combines 1-min microstructure, wave oscillator propagator, and Maritime Physical Supply Index (MPSI).
    """
    z_vwap = brent_feat.get("vwap_z", 0.0)
    ret_5m = brent_feat.get("ret_5m", 0.0)
    ret_10m = brent_feat.get("ret_10m", 0.0)
    w_dyn = brent_feat.get("wave_dynamics", {})
    z_mpsi = mpsi.get("z_maritime", 1.2)

    # Base technical score [-1.0, 1.0]
    base_mom = 0.40 * np.tanh(ret_5m / 0.35) + 0.35 * np.tanh(z_vwap / 1.5) + 0.25 * np.tanh(ret_10m / 0.6)

    # Maritime conditioning impact (Bullish chokepoint constriction lifts forward bias)
    # 1m: mainly microstructure + immediate shock
    # 10m: balanced technical momentum + maritime tightness
    # 30m & 1h: heavily driven by structural maritime physical deficit and backwardation roll
    prop_1m = w_dyn.get("z_pred_1m", z_vwap)
    prop_10m = w_dyn.get("z_pred_10m", z_vwap)
    prop_30m = w_dyn.get("z_pred_30m", z_vwap)
    prop_1h = w_dyn.get("z_pred_1h", z_vwap)

    score_1m = float(np.clip(0.60 * base_mom + 0.25 * np.tanh(prop_1m) + 0.15 * (z_mpsi / 3.0), -1.0, 1.0))
    score_10m = float(np.clip(0.45 * base_mom + 0.25 * np.tanh(prop_10m) + 0.30 * (z_mpsi / 3.0), -1.0, 1.0))
    score_30m = float(np.clip(0.30 * base_mom + 0.25 * np.tanh(prop_30m) + 0.45 * (z_mpsi / 3.0), -1.0, 1.0))
    score_1h = float(np.clip(0.20 * base_mom + 0.20 * np.tanh(prop_1h) + 0.60 * (z_mpsi / 3.0), -1.0, 1.0))

    def _to_signal(score: float, default_conf: float):
        p_up = float(np.clip(0.50 + 0.42 * score, 0.05, 0.95))
        conf = float(np.clip(default_conf + 0.15 * abs(score), 0.50, 0.98))
        if p_up >= 0.54:
            action = "STRONG_BUY" if p_up >= 0.68 else "BUY"
        elif p_up <= 0.46:
            action = "STRONG_SELL" if p_up <= 0.32 else "SELL"
        else:
            action = "NEUTRAL"
        return action, round(p_up, 4), round(conf, 4)

    act_1m, p_1m, conf_1m = _to_signal(score_1m, 0.72)
    act_10m, p_10m, conf_10m = _to_signal(score_10m, 0.75)
    act_30m, p_30m, conf_30m = _to_signal(score_30m, 0.78)
    act_1h, p_1h, conf_1h = _to_signal(score_1h, 0.82)

    probs = [p_1m, p_10m, p_30m, p_1h]
    avg_p = float(np.mean(probs))
    bull_c = sum(1 for p in probs if p > 0.53)
    bear_c = sum(1 for p in probs if p < 0.47)

    if bull_c == 4:
        alignment = "STRONG_BULLISH_ALIGNMENT (4/4 Horizons Up — Maritime Supply Pin)"
    elif bear_c == 4:
        alignment = "STRONG_BEARISH_ALIGNMENT (4/4 Horizons Down)"
    elif bull_c >= 3:
        alignment = "MODERATE_BULLISH_BIAS (3/4 Horizons Up)"
    elif bear_c >= 3:
        alignment = "MODERATE_BEARISH_BIAS (3/4 Horizons Down)"
    else:
        alignment = "DIVERGENT_CHOP / NEUTRAL"

    # Forward price estimations based on expected forward log-returns
    curr_price = float(brent_feat["price"])
    # Volatility scaling
    atr_pct = float((brent_feat.get("vwap_z", 0.0) * 0.002))
    drift_1m = (p_1m - 0.50) * 0.003
    drift_10m = (p_10m - 0.50) * 0.008
    drift_30m = (p_30m - 0.50) * 0.015
    drift_1h = (p_1h - 0.50) * 0.025

    pred_1m = round(curr_price * (1.0 + drift_1m), 2)
    pred_10m = round(curr_price * (1.0 + drift_10m), 2)
    pred_30m = round(curr_price * (1.0 + drift_30m), 2)
    pred_1h = round(curr_price * (1.0 + drift_1h), 2)

    return {
        "is_live_jev": False,
        "status": "calibrated_maritime_prior",
        "model_type": "MARITIME_TANKER_FORWARD_MODEL",
        "alignment": alignment,
        "average_prob_up": round(avg_p, 4),
        "maritime_supply_index": mpsi,
        "forward_projections": {
            "current_price": curr_price,
            "pred_1m": pred_1m,
            "pred_10m": pred_10m,
            "pred_30m": pred_30m,
            "pred_1h": pred_1h,
            "exp_1h_change_pct": round(((pred_1h / curr_price) - 1.0) * 100, 2)
        },
        "forward_1m": {
            "horizon": "1min forward (1 bar)",
            "action": act_1m,
            "prob_up": p_1m,
            "confidence": conf_1m,
            "projected_price": pred_1m
        },
        "forward_10m": {
            "horizon": "10min forward (10 bars)",
            "action": act_10m,
            "prob_up": p_10m,
            "confidence": conf_10m,
            "projected_price": pred_10m
        },
        "forward_30m": {
            "horizon": "30min forward (30 bars)",
            "action": act_30m,
            "prob_up": p_30m,
            "confidence": conf_30m,
            "projected_price": pred_30m
        },
        "forward_1h": {
            "horizon": "1h forward (60 bars)",
            "action": act_1h,
            "prob_up": p_1h,
            "confidence": conf_1h,
            "projected_price": pred_1h
        }
    }


def query_typesafe_jev_maritime_brent(
    prompt: str,
    brent_feat: dict,
    telemetry: dict,
    mpsi: dict,
    wti_feat: dict = None
) -> dict:
    """
    Dispatches query to TypeSafe Jev System One (jev-latest) with structured questions
    conditioned on shipping, chokepoint flows, crack spreads, and backwardation.
    """
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        return simulate_calibrated_maritime_brent_prior(brent_feat, telemetry, mpsi, wti_feat)

    payload = {
        "model": "jev-latest",
        "state": prompt,
        "questions": {
            "signal_1m_action": {
                "type": "choice",
                "instructions": "Predict the 1-minute forward trend direction for Brent Crude considering micro-flow and immediate chokepoint tightness.",
                "criteria": {
                    "STRONG_BUY": "Decisive upward price expansion expected in the next 1 minute",
                    "BUY": "Mild upward continuation expected in the next 1 minute",
                    "NEUTRAL": "Choppy or balanced price action near current level",
                    "SELL": "Mild downward drift expected in the next 1 minute",
                    "STRONG_SELL": "Sharp downward break expected in the next 1 minute"
                }
            },
            "prob_up_1m": {
                "type": "noul",
                "instructions": "Continuous probability (0.0 to 1.0) that Brent Crude closes higher in 1 minute."
            },
            "signal_10m_action": {
                "type": "choice",
                "instructions": "Predict the 10-minute forward trend direction for Brent Crude conditioned on maritime bottlenecks and VWAP.",
                "criteria": {
                    "STRONG_BUY": "Substantial upward expansion over next 10 minutes",
                    "BUY": "Bullish trend continuation over next 10 minutes",
                    "NEUTRAL": "Rangebound oscillation over next 10 minutes",
                    "SELL": "Bearish trend continuation over next 10 minutes",
                    "STRONG_SELL": "Accelerating liquidation over next 10 minutes"
                }
            },
            "prob_up_10m": {
                "type": "noul",
                "instructions": "Continuous probability (0.0 to 1.0) that Brent Crude closes higher in 10 minutes."
            },
            "signal_30m_action": {
                "type": "choice",
                "instructions": "Predict the 30-minute forward trend direction for Brent Crude conditioned on tanker rerouting and Asian demand pull.",
                "criteria": {
                    "STRONG_BUY": "Powerful 30m bull trend driven by physical inventory drain",
                    "BUY": "Steady upward appreciation over next 30 minutes",
                    "NEUTRAL": "Balanced or consolidating price action",
                    "SELL": "Downward mean-reversion over next 30 minutes",
                    "STRONG_SELL": "Steep physical liquidation or supply recovery"
                }
            },
            "prob_up_30m": {
                "type": "noul",
                "instructions": "Continuous probability (0.0 to 1.0) that Brent Crude closes higher in 30 minutes."
            },
            "signal_1h_action": {
                "type": "choice",
                "instructions": "Predict the 1-hour forward trend direction for Brent Crude conditioned on forward curve backwardation and middle distillate cracks.",
                "criteria": {
                    "STRONG_BUY": "High-conviction macro upward expansion over the next 1 hour",
                    "BUY": "Bullish upward drift supported by forward curve backwardation",
                    "NEUTRAL": "Neutral rangebound trade over the next 1 hour",
                    "SELL": "Bearish drift over the next 1 hour",
                    "STRONG_SELL": "Decisive breakdown over the next 1 hour"
                }
            },
            "prob_up_1h": {
                "type": "noul",
                "instructions": "Continuous probability (0.0 to 1.0) that Brent Crude closes higher in 1 hour."
            },
            "physical_regime_classification": {
                "type": "choice",
                "instructions": "Classify the predominant physical maritime and curve regime governing Brent forward pricing.",
                "criteria": {
                    "ACUTE_CHOKEPOINT_PIN": "Hormuz/Red Sea transit constrictions force steep backwardation premium",
                    "REFINERY_CRACK_PULL": "Elevated gasoil and jet cracks pull prompt crude feedstock demand",
                    "DESTOCKING_PULLBACK": "Offshore floating storage release temporarily softening prompt pricing",
                    "EQUILIBRIUM_FLOW": "Normalized shipping transit volume without acute chokepoint premiums"
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
        conf_1m = answers.get("signal_1m_action", {}).get("confidence", 0.72)
        p_1m = answers.get("prob_up_1m", {}).get("noul", 0.50)

        act_10m = answers.get("signal_10m_action", {}).get("choice", "NEUTRAL")
        conf_10m = answers.get("signal_10m_action", {}).get("confidence", 0.75)
        p_10m = answers.get("prob_up_10m", {}).get("noul", 0.50)

        act_30m = answers.get("signal_30m_action", {}).get("choice", "NEUTRAL")
        conf_30m = answers.get("signal_30m_action", {}).get("confidence", 0.78)
        p_30m = answers.get("prob_up_30m", {}).get("noul", 0.50)

        act_1h = answers.get("signal_1h_action", {}).get("choice", "NEUTRAL")
        conf_1h = answers.get("signal_1h_action", {}).get("confidence", 0.80)
        p_1h = answers.get("prob_up_1h", {}).get("noul", 0.50)

        probs = [p_1m, p_10m, p_30m, p_1h]
        avg_p = float(np.mean(probs))
        bull_c = sum(1 for p in probs if p > 0.53)
        bear_c = sum(1 for p in probs if p < 0.47)

        if bull_c == 4:
            alignment = "STRONG_BULLISH_ALIGNMENT (4/4 Horizons Up — Maritime Supply Pin)"
        elif bear_c == 4:
            alignment = "STRONG_BEARISH_ALIGNMENT (4/4 Horizons Down)"
        elif bull_c >= 3:
            alignment = "MODERATE_BULLISH_BIAS (3/4 Horizons Up)"
        elif bear_c >= 3:
            alignment = "MODERATE_BEARISH_BIAS (3/4 Horizons Down)"
        else:
            alignment = "DIVERGENT_CHOP / NEUTRAL"

        curr_price = float(brent_feat["price"])
        pred_1m = round(curr_price * (1.0 + (p_1m - 0.50) * 0.003), 2)
        pred_10m = round(curr_price * (1.0 + (p_10m - 0.50) * 0.008), 2)
        pred_30m = round(curr_price * (1.0 + (p_30m - 0.50) * 0.015), 2)
        pred_1h = round(curr_price * (1.0 + (p_1h - 0.50) * 0.025), 2)

        return {
            "is_live_jev": True,
            "status": "live_jev_api",
            "model_type": "MARITIME_TANKER_FORWARD_MODEL",
            "alignment": alignment,
            "average_prob_up": round(avg_p, 4),
            "maritime_supply_index": mpsi,
            "forward_projections": {
                "current_price": curr_price,
                "pred_1m": pred_1m,
                "pred_10m": pred_10m,
                "pred_30m": pred_30m,
                "pred_1h": pred_1h,
                "exp_1h_change_pct": round(((pred_1h / curr_price) - 1.0) * 100, 2)
            },
            "forward_1m": {
                "horizon": "1min forward (1 bar)",
                "action": act_1m,
                "prob_up": round(float(p_1m), 4),
                "confidence": round(float(conf_1m), 4),
                "projected_price": pred_1m
            },
            "forward_10m": {
                "horizon": "10min forward (10 bars)",
                "action": act_10m,
                "prob_up": round(float(p_10m), 4),
                "confidence": round(float(conf_10m), 4),
                "projected_price": pred_10m
            },
            "forward_30m": {
                "horizon": "30min forward (30 bars)",
                "action": act_30m,
                "prob_up": round(float(p_30m), 4),
                "confidence": round(float(conf_30m), 4),
                "projected_price": pred_30m
            },
            "forward_1h": {
                "horizon": "1h forward (60 bars)",
                "action": act_1h,
                "prob_up": round(float(p_1h), 4),
                "confidence": round(float(conf_1h), 4),
                "projected_price": pred_1h
            },
            "raw_response": data
        }
    except Exception as e:
        logger.warning(f"TypeSafe Jev maritime call failed: {e}. Falling back to calibrated prior.")
        return simulate_calibrated_maritime_brent_prior(brent_feat, telemetry, mpsi, wti_feat)


def compute_maritime_visual_analytics(
    brent_feat: dict,
    telemetry: dict,
    mpsi: dict,
    signals: dict
) -> dict:
    """
    Computes visual analytics data structures demonstrating explicitly how AIS data
    alters the Brent crude forward price estimate:
    1. Dollar-per-barrel attribution waterfall of the AIS physical disruption premium.
    2. Multi-horizon forward price trajectory comparing pure technical vs. AIS-conditioned paths.
    3. Chokepoint sensitivity curves (elasticity to Hormuz dark rate, Cape delays, Gasoil cracks).
    4. MPSI 4-pillar radar vector.
    """
    curr_p = float(brent_feat.get("price", 104.50))
    z_m = float(mpsi.get("z_maritime", 2.60))
    comps = mpsi.get("components", {})
    s_choke = float(comps.get("chokepoints_score", 3.0))
    s_crack = float(comps.get("refinery_crack_score", 2.13))
    s_stor = float(comps.get("storage_buffer_score", 2.27))
    s_back = float(comps.get("backwardation_score", 2.85))

    # Pure technical projected price at 1h (excluding AIS MPSI)
    ret_5m = float(brent_feat.get("ret_5m", 0.0))
    z_vwap = float(brent_feat.get("vwap_z", 0.0))
    base_mom = 0.40 * np.tanh(ret_5m / 0.35) + 0.35 * np.tanh(z_vwap / 1.5)
    tech_drift_1h = float(base_mom * 0.008)

    # AIS factor dollar contributions to the 1-hour forward price
    ais_total_premium = round(max(0.20, (z_m / 3.0) * 1.80), 2)

    w_choke = 0.38
    w_crack = 0.28
    w_stor = 0.18
    w_back = 0.16

    attrib_hormuz = round(ais_total_premium * w_choke * (s_choke / 3.0), 2)
    attrib_redsea_cape = round(ais_total_premium * w_stor * (s_stor / 3.0), 2)
    attrib_gasoil_crack = round(ais_total_premium * w_crack * (s_crack / 3.0), 2)
    attrib_backwardation = round(ais_total_premium * w_back * (s_back / 3.0), 2)
    total_physical_lift = round(attrib_hormuz + attrib_redsea_cape + attrib_gasoil_crack + attrib_backwardation, 2)

    # Trajectory comparison over 8 time steps [0, 1, 5, 10, 15, 30, 45, 60] minutes
    steps = [0, 1, 5, 10, 15, 30, 45, 60]
    time_labels = ["Prompt (t=0)", "+1m", "+5m", "+10m", "+15m", "+30m", "+45m", "+60m (1h)"]
    tech_trajectory = []
    maritime_trajectory = []
    physical_wedge = []
    upper_band = []
    lower_band = []

    p1m = signals.get("forward_1m", {}).get("projected_price", curr_p)
    p10m = signals.get("forward_10m", {}).get("projected_price", curr_p)
    p30m = signals.get("forward_30m", {}).get("projected_price", curr_p)
    p60m = signals.get("forward_1h", {}).get("projected_price", curr_p)

    for m in steps:
        if m == 0:
            p_tech = curr_p
            p_mar = curr_p
        elif m == 1:
            p_tech = round(curr_p * (1.0 + tech_drift_1h * 0.15), 2)
            p_mar = p1m
        elif m <= 10:
            ratio = (m - 1) / 9.0
            p_tech = round(curr_p * (1.0 + tech_drift_1h * 0.35 * ratio), 2)
            p_mar = round(p1m + (p10m - p1m) * ratio, 2)
        elif m <= 30:
            ratio = (m - 10) / 20.0
            p_tech = round(curr_p * (1.0 + tech_drift_1h * (0.35 + 0.35 * ratio)), 2)
            p_mar = round(p10m + (p30m - p10m) * ratio, 2)
        else:
            ratio = (m - 30) / 30.0
            p_tech = round(curr_p * (1.0 + tech_drift_1h * (0.70 + 0.30 * ratio)), 2)
            p_mar = round(p30m + (p60m - p30m) * ratio, 2)

        wedge = round(p_mar - p_tech, 2)
        vol_band = round(0.12 * np.sqrt(max(1, m)), 2)

        tech_trajectory.append(p_tech)
        maritime_trajectory.append(p_mar)
        physical_wedge.append(wedge)
        upper_band.append(round(p_mar + vol_band, 2))
        lower_band.append(round(p_mar - vol_band, 2))

    # Scenario sensitivity curves (What-if elasticity)
    hormuz_points = []
    for dark_pct in [10, 25, 40, 55, 70, 85, 95, 100]:
        price_delta = round((dark_pct - 35.0) / 35.0 * 0.75, 2)
        hormuz_points.append({"dark_rate_pct": dark_pct, "delta_usd": price_delta, "implied_1h_price": round(curr_p + price_delta, 2)})

    cape_points = []
    for delay in [0, 4, 8, 12, 16, 20, 24]:
        price_delta = round((delay - 0.0) / 12.0 * 0.65, 2)
        cape_points.append({"cape_delay_days": delay, "delta_usd": price_delta, "implied_1h_price": round(curr_p + price_delta, 2)})

    crack_points = []
    for crack in [15, 20, 25, 30, 35, 40]:
        price_delta = round((crack - 18.0) / 5.0 * 0.45, 2)
        crack_points.append({"gasoil_crack_usd": crack, "delta_usd": price_delta, "implied_1h_price": round(curr_p + price_delta, 2)})

    return {
        "attribution_waterfall": {
            "current_prompt_price": curr_p,
            "pure_technical_drift_usd": round(tech_drift_1h * curr_p, 2),
            "hormuz_constriction_lift_usd": attrib_hormuz,
            "redsea_cape_rerouting_lift_usd": attrib_redsea_cape,
            "gasoil_crack_refinery_lift_usd": attrib_gasoil_crack,
            "backwardation_roll_lift_usd": attrib_backwardation,
            "total_ais_physical_premium_usd": total_physical_lift,
            "implied_maritime_1h_price": round(curr_p + (tech_drift_1h * curr_p) + total_physical_lift, 2)
        },
        "trajectory_comparison": {
            "time_labels": time_labels,
            "steps_minutes": steps,
            "pure_technical_path": tech_trajectory,
            "maritime_ais_path": maritime_trajectory,
            "physical_supply_wedge_usd": physical_wedge,
            "upper_band_1sigma": upper_band,
            "lower_band_1sigma": lower_band
        },
        "scenario_elasticity": {
            "hormuz_sensitivity": hormuz_points,
            "cape_delay_sensitivity": cape_points,
            "gasoil_crack_sensitivity": crack_points
        },
        "radar_pillars": [
            {"pillar": "Chokepoints Constriction", "score": s_choke, "max_score": 3.0, "pct_tightness": round((s_choke/3.0)*100, 1), "detail": f"Hormuz 93.8% dark + Red Sea bypass"},
            {"pillar": "Refinery Margin Pull", "score": s_crack, "max_score": 3.0, "pct_tightness": round((s_crack/3.0)*100, 1), "detail": f"Gasoil crack ${telemetry.get('energy_state',{}).get('singapore_gasoil_crack_usd', 28.5):.2f}/bbl"},
            {"pillar": "Storage Depletion Buffer", "score": s_stor, "max_score": 3.0, "pct_tightness": round((s_stor/3.0)*100, 1), "detail": f"Floating storage {telemetry.get('energy_state',{}).get('fujairah_asean_floating_storage_mbbls', 18.2):.1f}M bbls"},
            {"pillar": "Forward Curve Inversion", "score": s_back, "max_score": 3.0, "pct_tightness": round((s_back/3.0)*100, 1), "detail": f"Prompt/M6 spread {telemetry.get('energy_state',{}).get('prompt_to_m6_spread', '+$9.70')}"}
        ]
    }


def generate_maritime_brent_signals(
    brent_df: pd.DataFrame,
    wti_df: Optional[pd.DataFrame] = None,
    loop_feedback: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for generating Maritime Tanker-Conditioned Brent Crude Forward Estimation.
    Fuses real-time 1m OHLCV bars, wave dynamics, and Oil_Tanker_Traffic_AntiGravity shipping telemetry.
    """
    brent_feat = compute_1m_features(brent_df)
    wti_feat = compute_1m_features(wti_df) if wti_df is not None else None

    # Ingest live maritime shipping telemetry
    telemetry = get_live_shipping_telemetry()
    mpsi = compute_maritime_physical_supply_index(telemetry)

    # Format state prompt
    state_prompt = format_maritime_brent_prompt(
        brent_feat=brent_feat,
        telemetry=telemetry,
        mpsi=mpsi,
        wti_feat=wti_feat,
        loop_feedback=loop_feedback
    )

    # Query TypeSafe Jev or calibrated mathematical prior
    signals = query_typesafe_jev_maritime_brent(
        prompt=state_prompt,
        brent_feat=brent_feat,
        telemetry=telemetry,
        mpsi=mpsi,
        wti_feat=wti_feat
    )

    visual_analytics = compute_maritime_visual_analytics(
        brent_feat=brent_feat,
        telemetry=telemetry,
        mpsi=mpsi,
        signals=signals
    )

    timestamp_str = (
        brent_df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(brent_df.index, pd.DatetimeIndex)
        else None
    )

    return {
        "asset_name": "Brent Crude (BRENT)",
        "model_variant": "MARITIME_TANKER_FORWARD_MODEL",
        "input_interval": "1m",
        "timestamp": timestamp_str,
        "features": brent_feat,
        "telemetry": telemetry,
        "mpsi": mpsi,
        "state_prompt": state_prompt,
        "signals": signals,
        "visual_analytics": visual_analytics
    }


if __name__ == "__main__":
    print("=== Testing Maritime-Conditioned Brent Crude Forward Estimation ===")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=75, freq="1min")
    prices = 104.50 + np.cumsum(np.random.normal(0.02, 0.08, len(dates)))
    synthetic_brent = pd.DataFrame({
        "Open": prices - 0.05,
        "High": prices + 0.12,
        "Low": prices - 0.10,
        "Close": prices,
        "Volume": np.random.randint(50, 400, len(dates))
    }, index=dates)

    res = generate_maritime_brent_signals(synthetic_brent)
    print("MPSI Summary:", res["mpsi"])
    print("Signals Summary:")
    print(json.dumps(res["signals"], indent=2))
