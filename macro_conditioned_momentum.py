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

# -------------------------------------------------------------------------
# Comprehensive Taxonomy & Guide for TypeSafe Jev System One Primitives:
# 1. Score: Continuous or ordered degree / conviction intensity
# 2. Choice: Categorical discrete decision among mutually exclusive options
# 3. Noul: Continuous truth likelihood / probability of a proposition (no separate confidence)
# -------------------------------------------------------------------------
JEV_PRIMITIVES_GUIDE = {
    "Choice": {
        "primitive": "Choice",
        "category": "Discrete Categorical Selection",
        "definition": (
            "Selects exactly one discrete option from a predefined, mutually exclusive set of categorical labels "
            "based on semantic criteria evaluated over current market state. Accompanied by a Confidence metric "
            "reflecting the concentration of probability on the winning option versus competing options."
        ),
        "forecasted_variables_referenced": {
            "forward_1m.action, forward_10m.action, forward_30m.action, forward_1h.action": {
                "type": "Choice",
                "options": ["STRONG_LONG", "LEAN_LONG", "NEUTRAL", "LEAN_SHORT", "STRONG_SHORT"],
                "description": "Tactical trading stance for each specific forward time horizon."
            },
            "nikkei_driver_regime (Nikkei 225)": {
                "type": "Choice",
                "options": ["FX_DOMINANT", "EQUITY_SENTIMENT_DOMINANT", "DUAL_REINFORCING", "DECOUPLED_CHOP"],
                "description": "Classifies whether Nikkei 225 is currently governed by USD/JPY FX moves, Wall Street equity sentiment, mutual compounding, or conflicting chop."
            },
            "cross_sectional_leadership": {
                "type": "Choice",
                "options": ["TOP_TIER_OUTPERFORMER", "MID_PACK_PARTICIPANT", "LAGGING_UNDERPERFORMER", "FLIGHT_TO_SAFETY_HEDGE"],
                "description": "Relative strength ranking across equities conditioned on yield and energy resilience."
            },
            "confidence": {
                "type": "Metric of Choice",
                "range": "0.00 to 1.00 (0% to 100%)",
                "description": "Quantifies the probability mass concentration on the winning choice relative to alternative choices."
            }
        }
    },
    "Score": {
        "primitive": "Score",
        "category": "Continuous / Ordered Dimension Degree",
        "definition": (
            "Measures degree or intensity along an ordered continuous or binned scale. Unlike a naive regression "
            "point estimate, Jev computes a calibrated probability distribution across ordered intervals, returning "
            "the probability-weighted expected position along that scale."
        ),
        "forecasted_variables_referenced": {
            "momentum_conviction_score": {
                "type": "Score",
                "range": "-100.0 to +100.0",
                "description": "Directional conviction intensity. Positive values (+1 to +100) indicate bullish conviction and momentum persistence; negative values (-1 to -100) indicate bearish conviction; values near 0 reflect neutral consolidation."
            },
            "fx_sensitivity_score (Nikkei 225)": {
                "type": "Score",
                "range": "0.0 to 100.0",
                "description": "Dynamic time-varying percentage weight of USD/JPY FX influence versus Global Equity Sentiment (0 = 100% US equity sentiment driven, 100 = 100% USD/JPY FX driven)."
            },
            "indicative_forward_target_prices (p_expected, p_lower, p_upper)": {
                "type": "Score",
                "range": "Price in currency units ($ / ¥ / pts)",
                "description": "Continuous expected forward price and volatility envelope derived from probability-weighted wave propagation."
            }
        }
    },
    "Noul": {
        "primitive": "Noul",
        "category": "Binary Truth Likelihood (Continuous Probability)",
        "definition": (
            "Evaluates the continuous probability that a specific proposition or condition holds true (P in [0.0, 1.0]). "
            "CRITICAL DISTINCTION: A Noul has NO separate confidence metric; the probability itself encapsulates uncertainty. "
            "P = 0.50 denotes complete epistemic neutrality / 50-50 tossup (maximum entropy), while P = 0.90 is high certainty "
            "of truth and P = 0.10 is high certainty of falsehood."
        ),
        "forecasted_variables_referenced": {
            "prob_up_1m, prob_up_10m, prob_up_30m, prob_up_1h (P_Up)": {
                "type": "Noul",
                "range": "0.0000 to 1.0000 (0% to 100%)",
                "description": "Exact continuous probability that the asset price will close strictly higher at the end of the specified forward horizon (1m, 10m, 30m, 1h) relative to current price."
            },
            "prob_fx_transmission (Nikkei 225)": {
                "type": "Noul",
                "range": "0.0000 to 1.0000 (0% to 100%)",
                "description": "Continuous probability that USD/JPY foreign exchange momentum will successfully transmit into Nikkei 225 price movement over the next 10 minutes."
            },
            "average_prob_up": {
                "type": "Noul",
                "range": "0.0000 to 1.0000",
                "description": "Composite cross-horizon directional probability across all 4 forward horizons."
            }
        }
    }
}

def decompose_nikkei_movements(
    ni_df: pd.DataFrame,
    usdjpy_df: pd.DataFrame = None,
    spx_df: pd.DataFrame = None,
    all_dfs: dict = None
) -> dict:
    """
    Rigorously decomposes Nikkei 225 price movements across multiple horizons
    (1m, 5m, 10m, 30m, 60m) into:
      1. USD/JPY FX Effect: Exporter profit translation & Yen carry-trade channel.
      2. Pure Equity Sentiment Effect: Global Wall Street / S&P 500 risk appetite,
         orthogonalized against USD/JPY to isolate pure equity sentiment.
      3. Domestic / Idiosyncratic Japan Alpha: Domestic BoJ policy, TSE microstructure,
         and domestic corporate earnings independent of macro factors.

    Also computes:
      - Rolling OLS Factor Betas: beta_fx, beta_equity_orthogonalized, beta_equity_raw.
      - Variance / ANOVA Decomposition (R^2, FX variance %, Pure Equity variance %, Idiosyncratic %).
      - Exact Additive Horizon Attribution (bps and %):
          Delta_NK(tau) == Delta_FX(tau) + Delta_PureEquity(tau) + Delta_Idiosyncratic(tau)
      - Quanto / CFD Denomination spread (Local JPY Nikkei return vs USD-denominated CFD).
      - TypeSafe Jev System One primitives (Choice, Score, Noul).
    """
    if usdjpy_df is None and all_dfs:
        for k in ["USD/JPY (USDJPY)", "USD_JPY", "JPY=X"]:
            if k in all_dfs and all_dfs[k] is not None:
                usdjpy_df = all_dfs[k]
                break
    if spx_df is None and all_dfs:
        for k in ["S&P 500 (SPX)", "SPX", "^GSPC"]:
            if k in all_dfs and all_dfs[k] is not None:
                spx_df = all_dfs[k]
                break

    if ni_df is None or len(ni_df) < 5:
        return {
            "has_decomposition": False,
            "primary_driver": "NEUTRAL",
            "primary_driver_desc": "Insufficient history for decomposition",
            "betas": {"beta_fx": 0.0, "beta_equity_ortho": 0.0, "beta_equity_raw": 0.0, "gamma_spx_fx": 0.0, "alpha_bps": 0.0},
            "variance_decomposition": {"fx_var_pct": 0.0, "pure_equity_var_pct": 0.0, "idiosyncratic_var_pct": 100.0, "r_squared": 0.0, "r_squared_pct": 0.0},
            "horizons": {},
            "scores": {"fx_contribution_score": 0.0, "pure_equity_sentiment_score": 0.0, "idiosyncratic_score": 0.0},
            "nouls": {"prob_fx_transmission": 0.50, "prob_equity_transmission": 0.50},
            "quanto_dynamics": {"usd_jpy_rate": 155.0, "exporter_margin_spread_5m_bps": 0.0}
        }

    ni_close = ni_df["Close"].values.astype(float)
    fx_close = usdjpy_df["Close"].values.astype(float) if usdjpy_df is not None and len(usdjpy_df) >= 5 else None
    spx_close = spx_df["Close"].values.astype(float) if spx_df is not None and len(spx_df) >= 5 else None

    n_bars = len(ni_close)
    if fx_close is not None:
        n_bars = min(n_bars, len(fx_close))
    if spx_close is not None:
        n_bars = min(n_bars, len(spx_close))

    w = max(5, min(n_bars, 45))
    ni_c = ni_close[-w:]
    r_ni = np.diff(ni_c) / (ni_c[:-1] + 1e-9)

    if fx_close is not None:
        fx_c = fx_close[-w:]
        r_fx = np.diff(fx_c) / (fx_c[:-1] + 1e-9)
    else:
        r_fx = np.zeros_like(r_ni)

    if spx_close is not None:
        sp_c = spx_close[-w:]
        r_sp = np.diff(sp_c) / (sp_c[:-1] + 1e-9)
    else:
        r_sp = np.zeros_like(r_ni)

    # 1. Orthogonalization of Pure Equity Sentiment against USD/JPY
    var_fx = float(np.var(r_fx))
    var_sp = float(np.var(r_sp))
    cov_sp_fx = float(np.cov(r_sp, r_fx)[0, 1]) if len(r_sp) > 2 and var_fx > 1e-12 else 0.0

    if var_fx > 1e-12:
        gamma_spx_fx = cov_sp_fx / var_fx
        r_sp_ortho = r_sp - gamma_spx_fx * r_fx
    else:
        gamma_spx_fx = 0.0
        r_sp_ortho = r_sp.copy()

    var_sp_ortho = float(np.var(r_sp_ortho))

    # 2. Factor Betas
    cov_ni_fx = float(np.cov(r_ni, r_fx)[0, 1]) if len(r_ni) > 2 and var_fx > 1e-12 else 0.0
    beta_fx = float(cov_ni_fx / var_fx) if var_fx > 1e-12 else 0.0

    cov_ni_sp_ortho = float(np.cov(r_ni, r_sp_ortho)[0, 1]) if len(r_ni) > 2 and var_sp_ortho > 1e-12 else 0.0
    beta_sp_ortho = float(cov_ni_sp_ortho / var_sp_ortho) if var_sp_ortho > 1e-12 else 0.0

    cov_ni_sp_raw = float(np.cov(r_ni, r_sp)[0, 1]) if len(r_ni) > 2 and var_sp > 1e-12 else 0.0
    beta_sp_raw = float(cov_ni_sp_raw / var_sp) if var_sp > 1e-12 else 0.0

    alpha_1m = float(np.mean(r_ni) - beta_fx * np.mean(r_fx) - beta_sp_ortho * np.mean(r_sp_ortho))

    # 3. Variance / ANOVA Decomposition
    var_ni = float(np.var(r_ni))
    var_fx_exp = (beta_fx ** 2) * var_fx
    var_sp_exp = (beta_sp_ortho ** 2) * var_sp_ortho
    var_exp_total = var_fx_exp + var_sp_exp
    var_denom = var_ni if var_ni > 1e-12 else (var_exp_total if var_exp_total > 1e-12 else 1e-12)

    r_squared = float(np.clip(var_exp_total / var_denom, 0.0, 1.0))
    share_var_fx = float(np.clip((var_fx_exp / var_denom) * 100.0, 0.0, 100.0))
    share_var_sp = float(np.clip((var_sp_exp / var_denom) * 100.0, 0.0, 100.0))
    if share_var_fx + share_var_sp > 100.0:
        tot = share_var_fx + share_var_sp
        share_var_fx = (share_var_fx / tot) * 95.0
        share_var_sp = (share_var_sp / tot) * 95.0
    share_var_idio = float(np.clip(100.0 - (share_var_fx + share_var_sp), 0.0, 100.0))

    # 4. Multi-Horizon Return Attribution
    horizons = {}
    h_spans = [1, 5, 10, 30, 60]
    for tau in h_spans:
        if len(ni_close) > tau:
            ret_ni_tau = float((ni_close[-1] / ni_close[-tau-1] - 1.0) * 100.0)
            ret_fx_tau = float((fx_close[-1] / fx_close[-tau-1] - 1.0) * 100.0) if fx_close is not None and len(fx_close) > tau else 0.0
            ret_sp_tau = float((spx_close[-1] / spx_close[-tau-1] - 1.0) * 100.0) if spx_close is not None and len(spx_close) > tau else 0.0

            ret_sp_ortho_tau = ret_sp_tau - gamma_spx_fx * ret_fx_tau

            fx_impact_pct = beta_fx * ret_fx_tau
            pure_equity_impact_pct = beta_sp_ortho * ret_sp_ortho_tau
            idio_impact_pct = ret_ni_tau - (fx_impact_pct + pure_equity_impact_pct)

            abs_sum = abs(fx_impact_pct) + abs(pure_equity_impact_pct) + abs(idio_impact_pct) + 1e-9
            share_fx = (abs(fx_impact_pct) / abs_sum) * 100.0
            share_eq = (abs(pure_equity_impact_pct) / abs_sum) * 100.0
            share_idio = (abs(idio_impact_pct) / abs_sum) * 100.0

            horizons[f"{tau}m"] = {
                "span_bars": tau,
                "ret_nikkei_pct": round(ret_ni_tau, 4),
                "ret_nikkei_bps": round(ret_ni_tau * 100.0, 1),
                "fx_impact_pct": round(fx_impact_pct, 4),
                "fx_impact_bps": round(fx_impact_pct * 100.0, 1),
                "pure_equity_impact_pct": round(pure_equity_impact_pct, 4),
                "pure_equity_impact_bps": round(pure_equity_impact_pct * 100.0, 1),
                "idiosyncratic_impact_pct": round(idio_impact_pct, 4),
                "idiosyncratic_impact_bps": round(idio_impact_pct * 100.0, 1),
                "share_fx_pct": round(share_fx, 1),
                "share_equity_pct": round(share_eq, 1),
                "share_idiosyncratic_pct": round(share_idio, 1)
            }

    # 5. Primary Driver Classification
    ref_h = horizons.get("5m") or horizons.get("1m") or {}
    fx_imp = ref_h.get("fx_impact_pct", 0.0)
    eq_imp = ref_h.get("pure_equity_impact_pct", 0.0)
    sh_fx = ref_h.get("share_fx_pct", 50.0)
    sh_eq = ref_h.get("share_equity_pct", 30.0)
    sh_idio = ref_h.get("share_idiosyncratic_pct", 20.0)

    if sh_fx >= 55.0:
        primary_driver = "FX_DOMINANT"
        driver_desc = f"USD/JPY FX currency transmission explains {sh_fx:.0f}% of Nikkei movement ({ref_h.get('fx_impact_bps', 0):+.1f} bps)."
    elif sh_eq >= 55.0:
        primary_driver = "PURE_EQUITY_SENTIMENT"
        driver_desc = f"Pure Global Equity Sentiment (Wall Street beta) explains {sh_eq:.0f}% of Nikkei movement ({ref_h.get('pure_equity_impact_bps', 0):+.1f} bps)."
    elif sh_idio >= 55.0:
        primary_driver = "DOMESTIC_IDIOSYNCRATIC"
        driver_desc = f"Domestic Japan alpha / Tokyo microstructure explains {sh_idio:.0f}% of movement ({ref_h.get('idiosyncratic_impact_bps', 0):+.1f} bps)."
    elif fx_imp * eq_imp > 0 and abs(fx_imp) > 0.01:
        primary_driver = "DUAL_AMPLIFYING"
        driver_desc = "USD/JPY currency momentum and US Equity Sentiment are mutually compounding Nikkei trend velocity."
    else:
        primary_driver = "CROSS_CANCELING"
        driver_desc = "USD/JPY FX and US Equity Sentiment are pulling in opposing directions, creating localized chop."

    fx_last_rate = float(fx_close[-1]) if fx_close is not None and len(fx_close) > 0 else 155.0
    exporter_margin_spread = float(fx_imp * 100.0)

    fx_score = float(np.clip(fx_imp * 200.0, -100.0, 100.0))
    eq_score = float(np.clip(eq_imp * 200.0, -100.0, 100.0))
    idio_score = float(np.clip(ref_h.get("idiosyncratic_impact_pct", 0.0) * 200.0, -100.0, 100.0))

    prob_fx_trans = float(np.clip(0.50 + 0.35 * (beta_fx / (abs(beta_fx) + 1.0)) * (1.0 if abs(fx_imp) > 0.01 else 0.5), 0.05, 0.95))
    prob_eq_trans = float(np.clip(0.50 + 0.35 * (beta_sp_ortho / (abs(beta_sp_ortho) + 1.0)) * (1.0 if abs(eq_imp) > 0.01 else 0.5), 0.05, 0.95))

    return {
        "has_decomposition": True,
        "primary_driver": primary_driver,
        "primary_driver_desc": driver_desc,
        "betas": {
            "beta_fx": round(beta_fx, 4),
            "beta_equity_ortho": round(beta_sp_ortho, 4),
            "beta_equity_raw": round(beta_sp_raw, 4),
            "gamma_spx_fx": round(gamma_spx_fx, 4),
            "alpha_1m_bps": round(alpha_1m * 10000.0, 2)
        },
        "variance_decomposition": {
            "fx_var_pct": round(share_var_fx, 1),
            "pure_equity_var_pct": round(share_var_sp, 1),
            "idiosyncratic_var_pct": round(share_var_idio, 1),
            "r_squared": round(r_squared, 4),
            "r_squared_pct": round(r_squared * 100.0, 1)
        },
        "horizons": horizons,
        "scores": {
            "fx_contribution_score": round(fx_score, 1),
            "pure_equity_sentiment_score": round(eq_score, 1),
            "idiosyncratic_score": round(idio_score, 1)
        },
        "nouls": {
            "prob_fx_transmission": round(prob_fx_trans, 4),
            "prob_equity_transmission": round(prob_eq_trans, 4)
        },
        "quanto_dynamics": {
            "usd_jpy_rate": round(fx_last_rate, 3),
            "nikkei_cfd_quote_ccy": "USD",
            "nikkei_local_ccy": "JPY",
            "exporter_margin_spread_5m_bps": round(exporter_margin_spread, 1)
        }
    }

def compute_nikkei_dynamic_fx_equity_regime(
    ni_df: pd.DataFrame,
    usdjpy_df: pd.DataFrame = None,
    spx_df: pd.DataFrame = None,
    all_dfs: dict = None
) -> dict:
    """
    Computes the dynamic time-varying beta and driver regime for Nikkei 225.
    Nikkei 225 is driven by USD/JPY FX dynamics (exporter currency translation, Yen carry trade)
    and Global Equity Sentiment (Wall Street S&P 500 risk appetite) at varying extent at different times.
    
    Returns:
    - regime: Choice primitive (FX_DOMINANT, EQUITY_SENTIMENT_DOMINANT, DUAL_REINFORCING, DECOUPLED_CHOP)
    - fx_weight, equity_weight: Dynamic weights w_fx and w_eq (w_fx + w_eq = 1.0)
    - fx_sensitivity_score: Score primitive (0 to 100)
    - prob_fx_transmission: Noul primitive (0.05 to 0.95)
    - rho_nikkei_fx, rho_nikkei_spx: Rolling correlations
    - dynamic_macro_impact: Directional scalar adjustment for Nikkei forward score
    - decomposition: Rigorous multi-horizon factor attribution & variance decomposition
    """
    if usdjpy_df is None and all_dfs:
        for k in ["USD/JPY (USDJPY)", "USD_JPY", "JPY=X"]:
            if k in all_dfs and all_dfs[k] is not None:
                usdjpy_df = all_dfs[k]
                break
    if spx_df is None and all_dfs:
        for k in ["S&P 500 (SPX)", "SPX", "^GSPC"]:
            if k in all_dfs and all_dfs[k] is not None:
                spx_df = all_dfs[k]
                break

    # Compute full econometric decomposition
    decomp = decompose_nikkei_movements(ni_df, usdjpy_df=usdjpy_df, spx_df=spx_df, all_dfs=all_dfs)

    # Fallback default values if USD/JPY data not available
    if usdjpy_df is None or len(usdjpy_df) < 5 or ni_df is None or len(ni_df) < 5:
        return {
            "has_fx_driver": False,
            "regime": "EQUITY_SENTIMENT_DOMINANT",
            "regime_desc": "USD/JPY feed offline; operating under baseline equity sentiment model.",
            "fx_weight": 0.35,
            "equity_weight": 0.65,
            "fx_sensitivity_score": 35.0,
            "prob_fx_transmission": 0.50,
            "rho_nikkei_fx": 0.65,
            "rho_nikkei_spx": 0.70,
            "dynamic_macro_impact": 0.0,
            "delta_fx": 0.0,
            "delta_eq": 0.0,
            "usdjpy_metrics": {},
            "spx_metrics": {},
            "decomposition": decomp
        }

    # Extract 1-minute close prices
    ni_close = ni_df["Close"].values.astype(float)
    fx_close = usdjpy_df["Close"].values.astype(float)
    
    # Calculate returns over rolling window
    k = min(len(ni_close), len(fx_close), 30)
    ni_rets = np.diff(ni_close[-k:]) / (ni_close[-k:-1] + 1e-9)
    fx_rets = np.diff(fx_close[-k:]) / (fx_close[-k:-1] + 1e-9)

    # Rolling correlation between Nikkei and USD/JPY
    if len(ni_rets) >= 5 and np.std(ni_rets) > 1e-7 and np.std(fx_rets) > 1e-7:
        rho_fx = float(np.corrcoef(ni_rets, fx_rets)[0, 1])
        rho_fx = float(np.nan_to_num(rho_fx, nan=0.65))
    else:
        rho_fx = 0.65

    # Rolling correlation between Nikkei and SPX (Equity Sentiment)
    if spx_df is not None and len(spx_df) >= 5:
        spx_close = spx_df["Close"].values.astype(float)
        k_sp = min(len(ni_close), len(spx_close), 30)
        sp_rets = np.diff(spx_close[-k_sp:]) / (spx_close[-k_sp:-1] + 1e-9)
        if len(sp_rets) >= 5 and np.std(sp_rets) > 1e-7 and np.std(ni_rets[-len(sp_rets):]) > 1e-7:
            rho_spx = float(np.corrcoef(ni_rets[-len(sp_rets):], sp_rets)[0, 1])
            rho_spx = float(np.nan_to_num(rho_spx, nan=0.70))
        else:
            rho_spx = 0.70
    else:
        spx_close = np.array([])
        rho_spx = 0.70

    # Compute USD/JPY micro-features
    fx_ret_1m = float((fx_close[-1] / fx_close[-2] - 1) * 100) if len(fx_close) >= 2 else 0.0
    fx_ret_5m = float((fx_close[-1] / fx_close[-6] - 1) * 100) if len(fx_close) >= 6 else fx_ret_1m
    fx_ret_10m = float((fx_close[-1] / fx_close[-11] - 1) * 100) if len(fx_close) >= 11 else fx_ret_5m

    fx_high = usdjpy_df["High"].values.astype(float)
    fx_low = usdjpy_df["Low"].values.astype(float)
    fx_tr = np.maximum(fx_high[-14:] - fx_low[-14:], 0.001)
    fx_atr = float(np.mean(fx_tr)) if len(fx_tr) > 0 else 0.01
    fx_vol = usdjpy_df["Volume"].values.astype(float) if "Volume" in usdjpy_df else np.ones(len(fx_close))
    fx_typical = (fx_high + fx_low + fx_close) / 3.0
    fx_vwap = float(np.sum(fx_typical[-14:] * fx_vol[-14:]) / (np.sum(fx_vol[-14:]) + 1e-9))
    fx_vwap_z = float((fx_close[-1] - fx_vwap) / (fx_atr + 1e-9))

    # Compute SPX micro-features
    if spx_df is not None and len(spx_df) >= 6:
        sp_c = spx_df["Close"].values.astype(float)
        sp_ret_5m = float((sp_c[-1] / sp_c[-6] - 1) * 100)
        sp_ret_10m = float((sp_c[-1] / sp_c[-11] - 1) * 100) if len(sp_c) >= 11 else sp_ret_5m
        sp_high = spx_df["High"].values.astype(float)
        sp_low = spx_df["Low"].values.astype(float)
        sp_atr = float(np.mean(np.maximum(sp_high[-14:] - sp_low[-14:], 0.1)))
        sp_vwap_z = float((sp_c[-1] - np.mean(sp_c[-14:])) / (sp_atr + 1e-9))
    else:
        sp_ret_5m = 0.0
        sp_ret_10m = 0.0
        sp_vwap_z = 0.0

    # Dynamic Time-Varying Beta & Allocation Weights
    i_fx = max(0.15, abs(rho_fx)) * (1.0 + min(2.5, abs(fx_ret_5m) * 2.0 + abs(fx_vwap_z) * 0.5))
    i_eq = max(0.15, abs(rho_spx)) * (1.0 + min(2.5, abs(sp_ret_5m) * 1.5 + abs(sp_vwap_z) * 0.4))
    
    total_i = i_fx + i_eq + 1e-9
    w_fx = float(np.clip(i_fx / total_i, 0.10, 0.90))
    w_eq = float(np.clip(1.0 - w_fx, 0.10, 0.90))

    # Regime Classification (Choice Primitive)
    fx_direction = 1 if fx_ret_5m > 0.01 else (-1 if fx_ret_5m < -0.01 else 0)
    sp_direction = 1 if sp_ret_5m > 0.02 else (-1 if sp_ret_5m < -0.02 else 0)

    if fx_direction != 0 and sp_direction != 0 and fx_direction == sp_direction:
        regime = "DUAL_REINFORCING"
        regime_desc = "USD/JPY currency momentum and US Equity Sentiment are mutually aligned, compounding Nikkei trend impulse."
    elif fx_direction != 0 and sp_direction != 0 and fx_direction != sp_direction:
        regime = "DECOUPLED_CHOP"
        regime_desc = "Opposing forces between USD/JPY currency dynamics and Wall Street equity sentiment causing chop and mean-reverting drag."
    elif w_fx >= 0.56:
        regime = "FX_DOMINANT"
        regime_desc = "USD/JPY currency momentum & Yen carry trade dynamics actively leading Nikkei 225 price action."
    else:
        regime = "EQUITY_SENTIMENT_DOMINANT"
        regime_desc = "Global Wall Street risk appetite & S&P 500 equity sentiment dominating over FX moves."

    # Quantitative FX and Equity transmission delta to Nikkei 225
    # Weakening Yen (USD/JPY up) boosts Nikkei exporter earnings; Yen surge (USD/JPY down) unwinds carry trade
    delta_fx = (0.35 * fx_ret_5m + 0.15 * fx_vwap_z) * np.sign(rho_fx if rho_fx != 0 else 1.0)
    delta_eq = (0.32 * sp_ret_5m + 0.12 * sp_vwap_z) * np.sign(rho_spx if rho_spx != 0 else 1.0)

    dynamic_macro_impact = float(w_fx * delta_fx + w_eq * delta_eq)

    # Score primitive: FX Sensitivity Score (0 to 100)
    fx_sensitivity_score = float(np.round(w_fx * 100.0, 1))

    # Noul primitive: Probability of FX Transmission (0.05 to 0.95)
    prob_fx_transmission = float(np.clip(0.50 + 0.30 * rho_fx + 0.20 * (w_fx - 0.50) * 2.0, 0.05, 0.95))

    return {
        "has_fx_driver": True,
        "regime": regime,
        "regime_desc": regime_desc,
        "fx_weight": round(w_fx, 4),
        "equity_weight": round(w_eq, 4),
        "fx_sensitivity_score": fx_sensitivity_score,
        "prob_fx_transmission": round(prob_fx_transmission, 4),
        "rho_nikkei_fx": round(rho_fx, 3),
        "rho_nikkei_spx": round(rho_spx, 3),
        "dynamic_macro_impact": round(dynamic_macro_impact, 4),
        "delta_fx": round(delta_fx, 4),
        "delta_eq": round(delta_eq, 4),
        "usdjpy_metrics": {
            "price": round(float(fx_close[-1]), 3),
            "ret_1m": round(fx_ret_1m, 3),
            "ret_5m": round(fx_ret_5m, 3),
            "ret_10m": round(fx_ret_10m, 3),
            "vwap_z": round(fx_vwap_z, 2)
        },
        "spx_metrics": {
            "price": round(float(spx_close[-1]), 2) if len(spx_close) > 0 else 0.0,
            "ret_5m": round(sp_ret_5m, 3),
            "ret_10m": round(sp_ret_10m, 3),
            "vwap_z": round(sp_vwap_z, 2)
        },
        "decomposition": decomp
    }

def format_macro_full_prompt(equity_name: str, eq_feat: dict, tnx_feat: dict, brent_feat: dict, wti_feat: dict, cs_context: dict = None, nikkei_dyn: dict = None) -> str:
    """
    Model A State Prompt: Equity Index state conditioned on 10Y Treasury Yield (TNX),
    Brent Crude, WTI Crude signals, Cross-Sectional Spreads & Rankings, nanoHUB Wave Mechanics,
    and Dynamic USD/JPY FX vs Equity Sentiment drivers for Nikkei 225.
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

    nikkei_block = ""
    if nikkei_dyn and nikkei_dyn.get("has_fx_driver"):
        fx_m = nikkei_dyn.get("usdjpy_metrics", {})
        sp_m = nikkei_dyn.get("spx_metrics", {})
        decomp = nikkei_dyn.get("decomposition", {})
        h5 = decomp.get("horizons", {}).get("5m", {})
        v_dec = decomp.get("variance_decomposition", {})
        betas = decomp.get("betas", {})
        nikkei_block = (
            f"--- Dynamic Nikkei 225 Macro Drivers & Factor Decomposition ---\n"
            f"USD/JPY FX (USD_JPY): Price={fx_m.get('price',0):.3f} | 1m Ret={fx_m.get('ret_1m',0):+.2f}%, 5m Ret={fx_m.get('ret_5m',0):+.2f}%, VWAP Z={fx_m.get('vwap_z',0):+.2f}σ | Rolling Corr ρ(NK, FX)={nikkei_dyn.get('rho_nikkei_fx',0):+.2f}\n"
            f"S&P 500 Equity Sentiment (SPX): Price={sp_m.get('price',0):,.2f} | 5m Ret={sp_m.get('ret_5m',0):+.2f}%, VWAP Z={sp_m.get('vwap_z',0):+.2f}σ | Rolling Corr ρ(NK, SPX)={nikkei_dyn.get('rho_nikkei_spx',0):+.2f}\n"
            f"Econometric 2-Factor Decomposition (5m Nikkei Move: {h5.get('ret_nikkei_pct', 0.0):+.2f}% / {h5.get('ret_nikkei_bps', 0.0):+.0f} bps):\n"
            f"  * USD/JPY FX Contribution: {h5.get('fx_impact_pct', 0.0):+.2f}% ({h5.get('fx_impact_bps', 0.0):+.0f} bps, {h5.get('share_fx_pct', 50):.0f}% share, β_FX={betas.get('beta_fx', 0.0):+.2f})\n"
            f"  * Pure Equity Sentiment:   {h5.get('pure_equity_impact_pct', 0.0):+.2f}% ({h5.get('pure_equity_impact_bps', 0.0):+.0f} bps, {h5.get('share_equity_pct', 30):.0f}% share, β_SPX_ortho={betas.get('beta_equity_ortho', 0.0):+.2f})\n"
            f"  * Idiosyncratic Japan Alpha:{h5.get('idiosyncratic_impact_pct', 0.0):+.2f}% ({h5.get('idiosyncratic_impact_bps', 0.0):+.0f} bps, {h5.get('share_idiosyncratic_pct', 20):.0f}% share)\n"
            f"Variance Attribution (ANOVA): FX={v_dec.get('fx_var_pct', 0):.1f}%, Pure Equity={v_dec.get('pure_equity_var_pct', 0):.1f}%, Idiosyncratic={v_dec.get('idiosyncratic_var_pct', 0):.1f}%, Model R²={v_dec.get('r_squared_pct', 0):.1f}%\n"
            f"Dynamic Time-Varying Beta Allocation: USD/JPY FX Weight = {nikkei_dyn.get('fx_weight',0.5)*100:.1f}%, Global Equity Sentiment Weight = {nikkei_dyn.get('equity_weight',0.5)*100:.1f}%\n"
            f"Active Driver Regime: {nikkei_dyn.get('regime','N/A')} ({nikkei_dyn.get('regime_desc','')})\n"
            f"Primary Movement Driver: {decomp.get('primary_driver', 'N/A')} ({decomp.get('primary_driver_desc', '')})\n"
            f"Dynamic Macro Impulse: Net Impact={nikkei_dyn.get('dynamic_macro_impact',0):+.4f} (FX Delta={nikkei_dyn.get('delta_fx',0):+.4f}, Eq Delta={nikkei_dyn.get('delta_eq',0):+.4f})\n"
        )

    return (
        f"Asset: {equity_name} (1-Minute Input Interval Bars)\n"
        f"Model Variant: MODEL A - FULL MACRO CROSS-ASSET (10Y Yield + Brent Crude + WTI Crude Conditioned)\n"
        f"Target Horizons: 1-Minute Forward (1 bar), 10-Minute Forward (10 bars), 30-Minute Forward (30 bars), 1-Hour Forward (60 bars)\n"
        f"--- Equity Microstructure ({equity_name}) ---\n"
        f"Latest Close Price: ${eq_feat['price']:,.2f}\n"
        f"VWAP Z-Score: {eq_feat['vwap_z']:+.2f} ATRs, RSI(14): {eq_feat['rsi_14']:.1f}, RVOL: {eq_feat['rvol']:.2f}x\n"
        f"Micro-Returns Curve: 1m={eq_feat['ret_1m']:+.2f}%, 3m={eq_feat['ret_3m']:+.2f}%, 5m={eq_feat['ret_5m']:+.2f}%, 10m={eq_feat['ret_10m']:+.2f}%, 30m={eq_feat['ret_30m']:+.2f}%, 60m={eq_feat.get('ret_60m', eq_feat['ret_30m']):+.2f}%\n"
        f"Realized 1-Min Autocorrelation: ρ_1 = {eq_feat.get('autocorr_1m', 0.0):+.2f}\n"
        f"Microstructure Dynamic Regime: {eq_feat.get('micro_regime', 'EQUILIBRIUM')}\n"
        f"Temporal Slopes: Micro-vs-Mid={eq_feat.get('slope_micro_vs_mid', 0.0):+.3f}%, Mid-vs-Macro={eq_feat.get('slope_mid_vs_macro', 0.0):+.3f}%\n"
        f"Temporal Term Structure Regime: {eq_feat.get('term_structure_type', 'N/A')}\n"
        f"{cs_block}"
        f"{w_block}"
        f"{nikkei_block}"
        f"--- Cross-Asset Macro Inputs ---\n"
        f"10Y Treasury Yield (TNX): ${tnx_feat['price']:.3f} | 5m Ret={tnx_feat['ret_5m']:+.2f}%, 10m Ret={tnx_feat['ret_10m']:+.2f}%, VWAP Z={tnx_feat['vwap_z']:+.2f}σ\n"
        f"Brent Crude Oil (BRENT): ${brent_feat['price']:.2f} | 5m Ret={brent_feat['ret_5m']:+.2f}%, 10m Ret={brent_feat['ret_10m']:+.2f}%, VWAP Z={brent_feat['vwap_z']:+.2f}σ\n"
        f"WTI Crude Oil (WTI): ${wti_feat['price']:.2f} | 5m Ret={wti_feat['ret_5m']:+.2f}%, 10m Ret={wti_feat['ret_10m']:+.2f}%, VWAP Z={wti_feat['vwap_z']:+.2f}σ\n"
        f"Macro Guidance: Rising yields (TNX) compress duration equity multiples. Surging crude oil (Brent/WTI) increases inflation drag. For Nikkei 225, USD/JPY FX weakening (rising) expands Japanese exporter margins, while Yen spikes trigger carry unwind liquidation; US equity sentiment sets the broader risk tone. In Mean-Reversion regimes, fade micro-bursts back to VWAP."
    )

def format_rate_only_prompt(equity_name: str, eq_feat: dict, tnx_feat: dict, cs_context: dict = None, nikkei_dyn: dict = None) -> str:
    """
    Model B State Prompt: Equity Index state conditioned on 10Y Treasury Yield (TNX) ONLY,
    excluding Brent and WTI crude oil, but including Dynamic USD/JPY FX for Nikkei 225.
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

    nikkei_block = ""
    if nikkei_dyn and nikkei_dyn.get("has_fx_driver"):
        fx_m = nikkei_dyn.get("usdjpy_metrics", {})
        sp_m = nikkei_dyn.get("spx_metrics", {})
        decomp = nikkei_dyn.get("decomposition", {})
        h5 = decomp.get("horizons", {}).get("5m", {})
        v_dec = decomp.get("variance_decomposition", {})
        betas = decomp.get("betas", {})
        nikkei_block = (
            f"--- Dynamic Nikkei 225 Macro Drivers & Factor Decomposition ---\n"
            f"USD/JPY FX (USD_JPY): Price={fx_m.get('price',0):.3f} | 1m Ret={fx_m.get('ret_1m',0):+.2f}%, 5m Ret={fx_m.get('ret_5m',0):+.2f}%, VWAP Z={fx_m.get('vwap_z',0):+.2f}σ | Rolling Corr ρ(NK, FX)={nikkei_dyn.get('rho_nikkei_fx',0):+.2f}\n"
            f"S&P 500 Equity Sentiment (SPX): Price={sp_m.get('price',0):,.2f} | 5m Ret={sp_m.get('ret_5m',0):+.2f}%, VWAP Z={sp_m.get('vwap_z',0):+.2f}σ | Rolling Corr ρ(NK, SPX)={nikkei_dyn.get('rho_nikkei_spx',0):+.2f}\n"
            f"Econometric 2-Factor Decomposition (5m Nikkei Move: {h5.get('ret_nikkei_pct', 0.0):+.2f}% / {h5.get('ret_nikkei_bps', 0.0):+.0f} bps):\n"
            f"  * USD/JPY FX Contribution: {h5.get('fx_impact_pct', 0.0):+.2f}% ({h5.get('fx_impact_bps', 0.0):+.0f} bps, {h5.get('share_fx_pct', 50):.0f}% share, β_FX={betas.get('beta_fx', 0.0):+.2f})\n"
            f"  * Pure Equity Sentiment:   {h5.get('pure_equity_impact_pct', 0.0):+.2f}% ({h5.get('pure_equity_impact_bps', 0.0):+.0f} bps, {h5.get('share_equity_pct', 30):.0f}% share, β_SPX_ortho={betas.get('beta_equity_ortho', 0.0):+.2f})\n"
            f"  * Idiosyncratic Japan Alpha:{h5.get('idiosyncratic_impact_pct', 0.0):+.2f}% ({h5.get('idiosyncratic_impact_bps', 0.0):+.0f} bps, {h5.get('share_idiosyncratic_pct', 20):.0f}% share)\n"
            f"Variance Attribution (ANOVA): FX={v_dec.get('fx_var_pct', 0):.1f}%, Pure Equity={v_dec.get('pure_equity_var_pct', 0):.1f}%, Idiosyncratic={v_dec.get('idiosyncratic_var_pct', 0):.1f}%, Model R²={v_dec.get('r_squared_pct', 0):.1f}%\n"
            f"Dynamic Time-Varying Beta Allocation: USD/JPY FX Weight = {nikkei_dyn.get('fx_weight',0.5)*100:.1f}%, Global Equity Sentiment Weight = {nikkei_dyn.get('equity_weight',0.5)*100:.1f}%\n"
            f"Active Driver Regime: {nikkei_dyn.get('regime','N/A')} ({nikkei_dyn.get('regime_desc','')})\n"
            f"Primary Movement Driver: {decomp.get('primary_driver', 'N/A')} ({decomp.get('primary_driver_desc', '')})\n"
            f"Dynamic Macro Impulse: Net Impact={nikkei_dyn.get('dynamic_macro_impact',0):+.4f} (FX Delta={nikkei_dyn.get('delta_fx',0):+.4f}, Eq Delta={nikkei_dyn.get('delta_eq',0):+.4f})\n"
        )

    return (
        f"Asset: {equity_name} (1-Minute Input Interval Bars)\n"
        f"Model Variant: MODEL B - RATE-ONLY MACRO (10Y Yield Conditioned, Excluding Crude Oil)\n"
        f"Target Horizons: 1-Minute Forward (1 bar), 10-Minute Forward (10 bars), 30-Minute Forward (30 bars), 1-Hour Forward (60 bars)\n"
        f"--- Equity Microstructure ({equity_name}) ---\n"
        f"Latest Close Price: ${eq_feat['price']:,.2f}\n"
        f"VWAP Z-Score: {eq_feat['vwap_z']:+.2f} ATRs, RSI(14): {eq_feat['rsi_14']:.1f}, RVOL: {eq_feat['rvol']:.2f}x\n"
        f"Micro-Returns Curve: 1m={eq_feat['ret_1m']:+.2f}%, 3m={eq_feat['ret_3m']:+.2f}%, 5m={eq_feat['ret_5m']:+.2f}%, 10m={eq_feat['ret_10m']:+.2f}%, 30m={eq_feat['ret_30m']:+.2f}%, 60m={eq_feat.get('ret_60m', eq_feat['ret_30m']):+.2f}%\n"
        f"Realized 1-Min Autocorrelation: ρ_1 = {eq_feat.get('autocorr_1m', 0.0):+.2f}\n"
        f"Microstructure Dynamic Regime: {eq_feat.get('micro_regime', 'EQUILIBRIUM')}\n"
        f"Temporal Slopes: Micro-vs-Mid={eq_feat.get('slope_micro_vs_mid', 0.0):+.3f}%, Mid-vs-Macro={eq_feat.get('slope_mid_vs_macro', 0.0):+.3f}%\n"
        f"Temporal Term Structure Regime: {eq_feat.get('term_structure_type', 'N/A')}\n"
        f"{cs_block}"
        f"{w_block}"
        f"{nikkei_block}"
        f"--- Rate Input (TNX Only) ---\n"
        f"10Y Treasury Yield (TNX): ${tnx_feat['price']:.3f} | 5m Ret={tnx_feat['ret_5m']:+.2f}%, 10m Ret={tnx_feat['ret_10m']:+.2f}%, VWAP Z={tnx_feat['vwap_z']:+.2f}σ\n"
        f"Macro Guidance: Evaluate equity forward trend strictly against interest rate duration sensitivity (10Y yield direction and volatility). For Nikkei 225, USD/JPY FX weakening (rising) expands Japanese exporter margins, while Yen spikes trigger carry unwind liquidation; US equity sentiment sets the broader risk tone. In Mean-Reversion regimes, fade micro-bursts back to VWAP."
    )

def simulate_macro_full_prior(equity_name: str, eq_feat: dict, tnx_feat: dict, brent_feat: dict, wti_feat: dict, nikkei_dyn: dict = None) -> dict:
    """
    Model A Calibrated Fallback Prior:
    Adjusts equity base score using 10Y Yield sensitivity + Brent/WTI Crude sensitivity + nanoHUB Wave Matrix Exp predictions.
    Dynamically conditions on microstructure autocorrelation (mean-reversion vs momentum persistence)
    and USD/JPY FX vs Equity Sentiment dynamic drivers for Nikkei 225.
    """
    ret_1m = eq_feat["ret_1m"]
    ret_3m = eq_feat["ret_3m"]
    ret_5m = eq_feat["ret_5m"]
    ret_10m = eq_feat["ret_10m"]
    ret_30m = eq_feat["ret_30m"]
    ret_60m = eq_feat.get("ret_60m", ret_30m)
    vwap_z = eq_feat["vwap_z"]
    autocorr = eq_feat.get("autocorr_1m", 0.0)

    w_dyn = eq_feat.get("wave_dynamics", {})
    z_w_1m = w_dyn.get("z_pred_1m", 0.0)
    z_w_10m = w_dyn.get("z_pred_10m", 0.0)
    z_w_30m = w_dyn.get("z_pred_30m", 0.0)
    z_w_1h = w_dyn.get("z_pred_1h", 0.0)

    # Base equity score incorporating wave exponential projections
    if autocorr < -0.05:
        base_score = -0.35 * ret_3m - 0.35 * ret_5m - 0.25 * vwap_z + 0.20 * (z_w_10m * 0.3)
    else:
        base_score = 0.20 * ret_1m + 0.25 * ret_3m + 0.25 * ret_5m + 0.20 * ret_10m + 0.25 * (vwap_z * 0.4) + 0.25 * (z_w_10m * 0.3)

    # 10Y Yield Sensitivity Factor
    rate_sens = 0.45 if "NDX" in equity_name else (0.40 if "RUT" in equity_name else 0.30)
    tnx_impact = - rate_sens * tnx_feat["ret_5m"] - 0.20 * tnx_feat["vwap_z"]

    # Energy Inflation Sensitivity Factor
    energy_ret = (brent_feat["ret_5m"] + wti_feat["ret_5m"]) / 2.0
    energy_z = (brent_feat["vwap_z"] + wti_feat["vwap_z"]) / 2.0
    energy_sens = 0.25 if "RUT" in equity_name else 0.20
    energy_impact = - energy_sens * energy_ret - 0.10 * energy_z

    # Dynamic USD/JPY FX and Equity Sentiment Transmission Factor for Nikkei 225
    nikkei_impact = nikkei_dyn.get("dynamic_macro_impact", 0.0) if nikkei_dyn else 0.0

    adjusted_score = base_score + tnx_impact + energy_impact + nikkei_impact

    # Probabilities for 1m, 10m, 30m, 1h incorporating specific wave horizon projections
    p_1m_input = -0.40 * ret_1m - 0.30 * vwap_z if autocorr < -0.05 else (0.40 * ret_1m + 0.30 * (z_w_1m * 0.3) + 0.10 * (tnx_impact + nikkei_impact * 0.5))
    prob_1m = float(np.clip(1.0 / (1.0 + np.exp(-(0.03 + 0.50 * p_1m_input))), 0.05, 0.95))
    prob_10m = float(np.clip(1.0 / (1.0 + np.exp(-(0.05 + 0.45 * (adjusted_score + 0.15 * z_w_10m + 0.05 * (tnx_impact + energy_impact) + 0.10 * nikkei_impact)))), 0.05, 0.95))
    prob_30m = float(np.clip(1.0 / (1.0 + np.exp(-(0.06 + 0.40 * (0.25 * ret_10m + 0.35 * ret_30m + 0.15 * z_w_30m + 0.10 * (tnx_impact + energy_impact + nikkei_impact))))), 0.05, 0.95))
    prob_1h = float(np.clip(1.0 / (1.0 + np.exp(-(0.08 + 0.38 * (0.20 * ret_30m + 0.35 * ret_60m + 0.15 * z_w_1h + 0.15 * (tnx_impact + energy_impact + nikkei_impact))))), 0.05, 0.95))

    def get_act(p):
        if p >= 0.56: return "STRONG_LONG"
        elif p >= 0.52: return "LEAN_LONG"
        elif p <= 0.44: return "STRONG_SHORT"
        elif p <= 0.48: return "LEAN_SHORT"
        return "NEUTRAL"

    probs = [prob_1m, prob_10m, prob_30m, prob_1h]
    avg_p = float(np.mean(probs))
    bull_c = sum(1 for p in probs if p > 0.52)
    bear_c = sum(1 for p in probs if p < 0.48)
    alignment = "STRONG_BULLISH_ALIGNMENT (4/4 Up)" if bull_c == 4 else ("STRONG_BEARISH_ALIGNMENT (4/4 Down)" if bear_c == 4 else ("MODERATE_BULLISH_BIAS (3/4 Up)" if bull_c >= 3 else ("MODERATE_BEARISH_BIAS (3/4 Down)" if bear_c >= 3 else "DIVERGENT_CHOP / NEUTRAL")))

    ip_data = compute_indicative_prices(eq_feat, prob_1m, prob_10m, prob_30m, prob_1h)

    # Compute overall Score primitive: Directional Momentum Conviction Score (-100 to +100)
    conviction_score = float(np.round(np.clip((prob_10m - 0.50) * 200.0 * 1.25 + 8.0 * adjusted_score, -100.0, 100.0), 1))

    f10m_extra = {"conviction_score": conviction_score}
    if nikkei_dyn and nikkei_dyn.get("has_fx_driver"):
        f10m_extra["nikkei_driver_regime"] = nikkei_dyn["regime"]
        f10m_extra["fx_sensitivity_score"] = nikkei_dyn["fx_sensitivity_score"]
        f10m_extra["prob_fx_transmission"] = nikkei_dyn["prob_fx_transmission"]

    return {
        "model_type": "MODEL_A_MACRO_FULL",
        "is_live_jev": False,
        "status": "simulation_mode",
        "alignment": alignment,
        "average_prob_up": round(avg_p, 4),
        "conviction_score": conviction_score,
        "forward_projections": ip_data["forward_projections"],
        "macro_impacts": {
            "rate_impact_tnx": round(tnx_impact, 4),
            "energy_impact_crude": round(energy_impact, 4),
            "nikkei_dynamic_impact": round(nikkei_impact, 4) if nikkei_dyn else 0.0
        },
        "nikkei_fx_equity_dynamics": nikkei_dyn if nikkei_dyn and nikkei_dyn.get("has_fx_driver") else None,
        "jev_primitives_guide": JEV_PRIMITIVES_GUIDE,
        "forward_1m": {"horizon": "1min forward", "action": get_act(prob_1m), "prob_up": round(prob_1m, 4), "confidence": 0.72, **ip_data["h1m"]},
        "forward_10m": {"horizon": "10min forward", "action": get_act(prob_10m), "prob_up": round(prob_10m, 4), "confidence": 0.76, **ip_data["h10m"], **f10m_extra},
        "forward_30m": {"horizon": "30min forward", "action": get_act(prob_30m), "prob_up": round(prob_30m, 4), "confidence": 0.78, **ip_data["h30m"]},
        "forward_1h": {"horizon": "1h forward", "action": get_act(prob_1h), "prob_up": round(prob_1h, 4), "confidence": 0.80, **ip_data["h1h"]}
    }

def simulate_rate_only_prior(equity_name: str, eq_feat: dict, tnx_feat: dict, nikkei_dyn: dict = None) -> dict:
    """
    Model B Calibrated Fallback Prior:
    Adjusts equity base score using 10Y Yield sensitivity ONLY (excluding Brent & WTI),
    incorporating USD/JPY FX dynamic drivers for Nikkei 225.
    """
    ret_1m = eq_feat["ret_1m"]
    ret_3m = eq_feat["ret_3m"]
    ret_5m = eq_feat["ret_5m"]
    ret_10m = eq_feat["ret_10m"]
    ret_30m = eq_feat["ret_30m"]
    ret_60m = eq_feat.get("ret_60m", ret_30m)
    vwap_z = eq_feat["vwap_z"]
    autocorr = eq_feat.get("autocorr_1m", 0.0)

    w_dyn = eq_feat.get("wave_dynamics", {})
    z_w_1m = w_dyn.get("z_pred_1m", 0.0)
    z_w_10m = w_dyn.get("z_pred_10m", 0.0)
    z_w_30m = w_dyn.get("z_pred_30m", 0.0)
    z_w_1h = w_dyn.get("z_pred_1h", 0.0)

    # Base equity score incorporating wave exponential projections
    if autocorr < -0.05:
        base_score = -0.35 * ret_3m - 0.35 * ret_5m - 0.25 * vwap_z + 0.20 * (z_w_10m * 0.3)
    else:
        base_score = 0.20 * ret_1m + 0.25 * ret_3m + 0.25 * ret_5m + 0.20 * ret_10m + 0.25 * (vwap_z * 0.4) + 0.25 * (z_w_10m * 0.3)

    # 10Y Yield Sensitivity Factor ONLY
    rate_sens = 0.50 if "NDX" in equity_name else (0.45 if "RUT" in equity_name else 0.35)
    tnx_impact = - rate_sens * tnx_feat["ret_5m"] - 0.25 * tnx_feat["vwap_z"]

    # Dynamic USD/JPY FX and Equity Sentiment Transmission Factor for Nikkei 225
    nikkei_impact = nikkei_dyn.get("dynamic_macro_impact", 0.0) if nikkei_dyn else 0.0

    adjusted_score = base_score + tnx_impact + nikkei_impact

    # Probabilities for 1m, 10m, 30m, 1h incorporating specific wave horizon projections
    p_1m_input = -0.40 * ret_1m - 0.30 * vwap_z if autocorr < -0.05 else (0.40 * ret_1m + 0.30 * (z_w_1m * 0.3) + 0.10 * (tnx_impact + nikkei_impact * 0.5))
    prob_1m = float(np.clip(1.0 / (1.0 + np.exp(-(0.03 + 0.50 * p_1m_input))), 0.05, 0.95))
    prob_10m = float(np.clip(1.0 / (1.0 + np.exp(-(0.05 + 0.45 * (adjusted_score + 0.15 * z_w_10m + 0.10 * nikkei_impact)))), 0.05, 0.95))
    prob_30m = float(np.clip(1.0 / (1.0 + np.exp(-(0.06 + 0.40 * (0.25 * ret_10m + 0.35 * ret_30m + 0.15 * z_w_30m + 0.10 * (tnx_impact + nikkei_impact))))), 0.05, 0.95))
    prob_1h = float(np.clip(1.0 / (1.0 + np.exp(-(0.08 + 0.38 * (0.20 * ret_30m + 0.35 * ret_60m + 0.15 * z_w_1h + 0.15 * (tnx_impact + nikkei_impact))))), 0.05, 0.95))

    def get_act(p):
        if p >= 0.56: return "STRONG_LONG"
        elif p >= 0.52: return "LEAN_LONG"
        elif p <= 0.44: return "STRONG_SHORT"
        elif p <= 0.48: return "LEAN_SHORT"
        return "NEUTRAL"

    probs = [prob_1m, prob_10m, prob_30m, prob_1h]
    avg_p = float(np.mean(probs))
    bull_c = sum(1 for p in probs if p > 0.52)
    bear_c = sum(1 for p in probs if p < 0.48)
    alignment = "STRONG_BULLISH_ALIGNMENT (4/4 Up)" if bull_c == 4 else ("STRONG_BEARISH_ALIGNMENT (4/4 Down)" if bear_c == 4 else ("MODERATE_BULLISH_BIAS (3/4 Up)" if bull_c >= 3 else ("MODERATE_BEARISH_BIAS (3/4 Down)" if bear_c >= 3 else "DIVERGENT_CHOP / NEUTRAL")))

    ip_data = compute_indicative_prices(eq_feat, prob_1m, prob_10m, prob_30m, prob_1h)

    # Compute overall Score primitive: Directional Momentum Conviction Score (-100 to +100)
    conviction_score = float(np.round(np.clip((prob_10m - 0.50) * 200.0 * 1.25 + 8.0 * adjusted_score, -100.0, 100.0), 1))

    f10m_extra = {"conviction_score": conviction_score}
    if nikkei_dyn and nikkei_dyn.get("has_fx_driver"):
        f10m_extra["nikkei_driver_regime"] = nikkei_dyn["regime"]
        f10m_extra["fx_sensitivity_score"] = nikkei_dyn["fx_sensitivity_score"]
        f10m_extra["prob_fx_transmission"] = nikkei_dyn["prob_fx_transmission"]

    return {
        "model_type": "MODEL_B_RATE_ONLY",
        "is_live_jev": False,
        "status": "simulation_mode",
        "alignment": alignment,
        "average_prob_up": round(avg_p, 4),
        "conviction_score": conviction_score,
        "forward_projections": ip_data["forward_projections"],
        "macro_impacts": {
            "rate_impact_tnx": round(tnx_impact, 4),
            "energy_impact_crude": 0.0,
            "nikkei_dynamic_impact": round(nikkei_impact, 4) if nikkei_dyn else 0.0
        },
        "nikkei_fx_equity_dynamics": nikkei_dyn if nikkei_dyn and nikkei_dyn.get("has_fx_driver") else None,
        "jev_primitives_guide": JEV_PRIMITIVES_GUIDE,
        "forward_1m": {"horizon": "1min forward", "action": get_act(prob_1m), "prob_up": round(prob_1m, 4), "confidence": 0.72, **ip_data["h1m"]},
        "forward_10m": {"horizon": "10min forward", "action": get_act(prob_10m), "prob_up": round(prob_10m, 4), "confidence": 0.76, **ip_data["h10m"], **f10m_extra},
        "forward_30m": {"horizon": "30min forward", "action": get_act(prob_30m), "prob_up": round(prob_30m, 4), "confidence": 0.78, **ip_data["h30m"]},
        "forward_1h": {"horizon": "1h forward", "action": get_act(prob_1h), "prob_up": round(prob_1h, 4), "confidence": 0.80, **ip_data["h1h"]}
    }

def query_typesafe_jev_macro(state_prompt: str, model_type: str, fallback_func, eq_feat: dict = None, is_nikkei: bool = False, nikkei_dyn: dict = None) -> dict:
    """Queries TypeSafe Jev System One API with macro-conditioned state representation and typed primitives."""
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        return fallback_func()

    questions = {
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
        "momentum_conviction_score": {
            "type": "score",
            "instructions": f"Rate overall directional momentum conviction under {model_type} from extreme bearish to extreme bullish.",
            "criteria": [
                "Extreme Bearish Momentum / Severe Downside (-100)",
                "Moderate Bearish Lean / Downside Continuation (-50)",
                "Neutral Equilibrium / Mean-Reverting Consolidation (0)",
                "Moderate Bullish Lean / Upside Continuation (+50)",
                "Extreme Bullish Momentum / Strong Upside Breakout (+100)"
            ]
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

    if is_nikkei:
        questions["nikkei_driver_regime"] = {
            "type": "choice",
            "instructions": "Determine whether Nikkei 225 forward momentum is primarily driven by USD/JPY FX currency momentum or US/Global Equity Sentiment.",
            "criteria": {
                "FX_DOMINANT": "USD/JPY FX currency moves and Yen carry trade unwinding/expansion are the primary leading driver of Nikkei 225 price action.",
                "EQUITY_SENTIMENT_DOMINANT": "Global equity risk appetite (S&P 500 / Nasdaq trend) dominates over FX dynamics.",
                "DUAL_REINFORCING": "USD/JPY FX and US equity sentiment are mutually aligned in the same direction, magnifying Nikkei momentum.",
                "DECOUPLED_CHOP": "Conflicting signals between USD/JPY FX and US equities resulting in choppy consolidation or range-bound mean-reversion."
            }
        }
        questions["fx_sensitivity_score"] = {
            "type": "score",
            "instructions": "Rate the degree to which USD/JPY currency dynamics are driving Nikkei 225 tactical direction over global equity sentiment.",
            "criteria": [
                "Purely Equity Sentiment Driven (0% FX, 100% S&P 500 / Global Risk)",
                "Primarily Equity Sentiment Driven (25% FX, 75% Global Risk)",
                "Balanced Dual Transmission (50% FX, 50% Global Risk)",
                "Primarily USD/JPY Currency Driven (75% FX, 25% Global Risk)",
                "Dominantly USD/JPY Currency Driven (100% FX, Yen Carry / Exporter Dominance)"
            ]
        }
        questions["prob_fx_transmission"] = {
            "type": "noul",
            "instructions": "What is the continuous probability that USD/JPY momentum will directly transmit into Nikkei 225 forward price return over the next 10 minutes?"
        }

    payload = {
        "model": "jev-latest",
        "state": state_prompt,
        "questions": questions
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

        # Compute empirical macro fallback prior for Bayesian calibration
        fb_res = fallback_func()

        raw_act_1m = answers.get("signal_1m_action", {}).get("choice", "NEUTRAL")
        conf_1m = answers.get("signal_1m_action", {}).get("confidence", 0.70)
        raw_p_1m = answers.get("prob_up_1m", {}).get("noul", 0.50)
        p_1m = float(np.clip(0.40 * float(raw_p_1m) + 0.60 * float(fb_res["forward_1m"]["prob_up"]), 0.05, 0.95))

        raw_act_10m = answers.get("signal_10m_action", {}).get("choice", "NEUTRAL")
        conf_10m = answers.get("signal_10m_action", {}).get("confidence", 0.72)
        raw_p_10m = answers.get("prob_up_10m", {}).get("noul", 0.50)
        p_10m = float(np.clip(0.40 * float(raw_p_10m) + 0.60 * float(fb_res["forward_10m"]["prob_up"]), 0.05, 0.95))

        raw_act_30m = answers.get("signal_30m_action", {}).get("choice", "NEUTRAL")
        conf_30m = answers.get("signal_30m_action", {}).get("confidence", 0.75)
        raw_p_30m = answers.get("prob_up_30m", {}).get("noul", 0.50)
        p_30m = float(np.clip(0.40 * float(raw_p_30m) + 0.60 * float(fb_res["forward_30m"]["prob_up"]), 0.05, 0.95))

        raw_act_1h = answers.get("signal_1h_action", {}).get("choice", "NEUTRAL")
        conf_1h = answers.get("signal_1h_action", {}).get("confidence", 0.78)
        raw_p_1h = answers.get("prob_up_1h", {}).get("noul", 0.50)
        p_1h = float(np.clip(0.40 * float(raw_p_1h) + 0.60 * float(fb_res["forward_1h"]["prob_up"]), 0.05, 0.95))

        raw_conv = answers.get("momentum_conviction_score", {}).get("score")
        if raw_conv is not None:
            # Score 0..4 mapped to -100..+100
            mapped_conv = (float(raw_conv) - 2.0) * 50.0
            conviction_score = float(np.round(0.40 * mapped_conv + 0.60 * float(fb_res.get("conviction_score", 0.0)), 1))
        else:
            conviction_score = fb_res.get("conviction_score", 0.0)

        def cal_act(raw, p):
            if p >= 0.56: return "STRONG_LONG"
            elif p >= 0.52: return "LEAN_LONG"
            elif p <= 0.44: return "STRONG_SHORT"
            elif p <= 0.48: return "LEAN_SHORT"
            return raw if raw != "NEUTRAL" else "NEUTRAL"

        act_1m = cal_act(raw_act_1m, p_1m)
        act_10m = cal_act(raw_act_10m, p_10m)
        act_30m = cal_act(raw_act_30m, p_30m)
        act_1h = cal_act(raw_act_1h, p_1h)

        probs = [p_1m, p_10m, p_30m, p_1h]
        avg_p = float(np.mean(probs))
        bull_c = sum(1 for p in probs if p > 0.52)
        bear_c = sum(1 for p in probs if p < 0.48)
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

        f10m_extra = {"conviction_score": conviction_score}
        nikkei_res = fb_res.get("nikkei_fx_equity_dynamics")
        if is_nikkei and nikkei_res:
            nik_choice = answers.get("nikkei_driver_regime", {}).get("choice", nikkei_res.get("regime"))
            raw_nik_score = answers.get("fx_sensitivity_score", {}).get("score")
            nik_score = float(np.round(float(raw_nik_score) * 25.0, 1)) if raw_nik_score is not None else nikkei_res.get("fx_sensitivity_score", 50.0)
            nik_noul = answers.get("prob_fx_transmission", {}).get("noul", nikkei_res.get("prob_fx_transmission", 0.50))
            nikkei_res["regime"] = nik_choice
            nikkei_res["fx_sensitivity_score"] = float(nik_score)
            nikkei_res["prob_fx_transmission"] = float(nik_noul)
            f10m_extra["nikkei_driver_regime"] = nik_choice
            f10m_extra["fx_sensitivity_score"] = float(nik_score)
            f10m_extra["prob_fx_transmission"] = float(nik_noul)

        return {
            "model_type": model_type,
            "is_live_jev": True,
            "status": "live_jev_api",
            "alignment": alignment,
            "average_prob_up": round(avg_p, 4),
            "conviction_score": conviction_score,
            "nikkei_fx_equity_dynamics": nikkei_res,
            "jev_primitives_guide": JEV_PRIMITIVES_GUIDE,
            **ip_extra,
            "forward_1m": {"horizon": "1min forward", "action": act_1m, "prob_up": round(float(p_1m), 4), "confidence": round(float(conf_1m), 4), **h1m_ext},
            "forward_10m": {"horizon": "10min forward", "action": act_10m, "prob_up": round(float(p_10m), 4), "confidence": round(float(conf_10m), 4), **h10m_ext, **f10m_extra},
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
    fx_ret = feats.get("USD/JPY (USDJPY)", {}).get("ret_5m", 0.0)

    tech_spread = float(np.round(ndx_ret - spx_ret, 3))
    beta_spread = float(np.round(rut_ret - spx_ret, 3))
    asia_spread = float(np.round(ni_ret - spx_ret, 3))

    rate_equity_spread = float(np.round(tnx_ret - spx_ret, 3))
    energy_equity_spread = float(np.round(((brent_ret + wti_ret) / 2.0) - spx_ret, 3))
    fx_equity_spread = float(np.round(fx_ret - spx_ret, 3))

    return {
        "equity_ranks": equity_ranks,
        "tech_spread": tech_spread,
        "beta_spread": beta_spread,
        "asia_spread": asia_spread,
        "rate_equity_spread": rate_equity_spread,
        "energy_equity_spread": energy_equity_spread,
        "fx_equity_spread": fx_equity_spread
    }

def generate_macro_full_signals(eq_df: pd.DataFrame, tnx_df: pd.DataFrame, brent_df: pd.DataFrame, wti_df: pd.DataFrame, equity_name: str = "S&P 500 (SPX)", all_dfs: dict = None, loop_feedback: str = None) -> dict:
    """
    Model A Generator: Computes equity trend signals conditioned on 10Y Yield, Brent, WTI, Cross-Sectional Spreads,
    and Dynamic USD/JPY FX vs Equity Sentiment drivers for Nikkei 225.
    """
    eq_feat = compute_1m_features(eq_df)
    tnx_feat = compute_1m_features(tnx_df)
    brent_feat = compute_1m_features(brent_df)
    wti_feat = compute_1m_features(wti_df)

    is_nikkei = "Nikkei" in equity_name or "NI225" in equity_name
    nikkei_dyn = compute_nikkei_dynamic_fx_equity_regime(eq_df, all_dfs=all_dfs) if is_nikkei else None

    cs_context = compute_cross_sectional_context(all_dfs) if all_dfs else None
    prompt = format_macro_full_prompt(equity_name, eq_feat, tnx_feat, brent_feat, wti_feat, cs_context, nikkei_dyn=nikkei_dyn)
    if loop_feedback:
        prompt += f"\n--- Empirical Closed-Loop Feedback (Walk-Forward Autotuning) ---\n{loop_feedback}\n"

    fallback = lambda: simulate_macro_full_prior(equity_name, eq_feat, tnx_feat, brent_feat, wti_feat, nikkei_dyn=nikkei_dyn)
    signals = query_typesafe_jev_macro(prompt, "MODEL_A_MACRO_FULL", fallback, eq_feat=eq_feat, is_nikkei=is_nikkei, nikkei_dyn=nikkei_dyn)

    return {
        "model_variant": "MODEL_A_MACRO_FULL",
        "description": "Affected by 10Y Yield (TNX) + Brent Crude + WTI Crude + Cross-Sectional Spreads" + (" + Dynamic USD/JPY FX" if is_nikkei else ""),
        "equity_name": equity_name,
        "cross_sectional_context": cs_context,
        "nikkei_fx_equity_dynamics": nikkei_dyn,
        "jev_primitives_guide": JEV_PRIMITIVES_GUIDE,
        "conviction_score": signals.get("conviction_score"),
        "signals": signals,
        "state_prompt": prompt
    }

def generate_rate_only_signals(eq_df: pd.DataFrame, tnx_df: pd.DataFrame, equity_name: str = "S&P 500 (SPX)", all_dfs: dict = None, loop_feedback: str = None) -> dict:
    """
    Model B Generator: Computes equity trend signals conditioned on 10Y Yield ONLY + Cross-Sectional Spreads,
    and Dynamic USD/JPY FX vs Equity Sentiment drivers for Nikkei 225.
    """
    eq_feat = compute_1m_features(eq_df)
    tnx_feat = compute_1m_features(tnx_df)

    is_nikkei = "Nikkei" in equity_name or "NI225" in equity_name
    nikkei_dyn = compute_nikkei_dynamic_fx_equity_regime(eq_df, all_dfs=all_dfs) if is_nikkei else None

    cs_context = compute_cross_sectional_context(all_dfs) if all_dfs else None
    prompt = format_rate_only_prompt(equity_name, eq_feat, tnx_feat, cs_context, nikkei_dyn=nikkei_dyn)
    if loop_feedback:
        prompt += f"\n--- Empirical Closed-Loop Feedback (Walk-Forward Autotuning) ---\n{loop_feedback}\n"

    fallback = lambda: simulate_rate_only_prior(equity_name, eq_feat, tnx_feat, nikkei_dyn=nikkei_dyn)
    signals = query_typesafe_jev_macro(prompt, "MODEL_B_RATE_ONLY", fallback, eq_feat=eq_feat, is_nikkei=is_nikkei, nikkei_dyn=nikkei_dyn)

    return {
        "model_variant": "MODEL_B_RATE_ONLY",
        "description": "Affected by 10Y Yield (TNX) ONLY + Cross-Sectional Spreads" + (" + Dynamic USD/JPY FX" if is_nikkei else ""),
        "equity_name": equity_name,
        "cross_sectional_context": cs_context,
        "nikkei_fx_equity_dynamics": nikkei_dyn,
        "jev_primitives_guide": JEV_PRIMITIVES_GUIDE,
        "conviction_score": signals.get("conviction_score"),
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
