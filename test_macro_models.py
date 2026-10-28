"""
Validation Suite for Macro-Conditioned Momentum Models:
- Model A (Full Macro with Yield + Brent + WTI)
- Model B (Rate-Only with 10Y Yield ONLY)
- Dynamic Nikkei 225 USD/JPY FX vs Equity Sentiment Driver Engine
- TypeSafe Jev System One Output Variables: Score, Choice, and Noul
- Institutional OANDA CFD Feed Priority

Author: Quantitative Trader & Mathematician
"""

import os
import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from macro_conditioned_momentum import (
    generate_macro_full_signals,
    generate_rate_only_signals,
    compute_nikkei_dynamic_fx_equity_regime,
    decompose_nikkei_movements,
    JEV_PRIMITIVES_GUIDE
)

def run_tests():
    print("=" * 110)
    print("   MODEL A, MODEL B, NIKKEI FX/EQUITY DYNAMICS & JEV PRIMITIVES VERIFICATION SUITE")
    print("=" * 110)

    dates = pd.date_range(end=pd.Timestamp.now(), periods=80, freq="1min")
    np.random.seed(123)

    make_series = lambda base, drift, vol: pd.DataFrame({
        "Open": base + np.cumsum(np.random.normal(drift, vol, 80)),
        "High": base + np.cumsum(np.random.normal(drift, vol, 80)) + 0.3,
        "Low": base + np.cumsum(np.random.normal(drift, vol, 80)) - 0.3,
        "Close": base + np.cumsum(np.random.normal(drift, vol, 80)),
        "Volume": np.random.randint(200, 1500, 80)
    }, index=dates)

    sp_df = make_series(5800.0, 0.05, 0.3)
    ndx_df = make_series(18500.0, 0.08, 1.0)
    rut_df = make_series(2200.0, -0.02, 0.2)
    ni_df = make_series(38000.0, 0.1, 2.0)

    tnx_df = make_series(3.95, 0.015, 0.005) # TNX surging upward
    brent_df = make_series(75.0, 0.2, 0.1)  # Brent surging upward
    wti_df = make_series(71.0, 0.2, 0.1)    # WTI surging upward
    usdjpy_df = make_series(157.0, 0.03, 0.02) # USD/JPY weakening Yen

    equities = {
        "S&P 500 (SPX)": sp_df,
        "Nasdaq 100 (NDX)": ndx_df,
        "Russell 2000 (RUT)": rut_df,
        "Nikkei 225 (NI225)": ni_df
    }

    all_dfs = {
        **equities,
        "10Y Treasury (TNX)": tnx_df,
        "Brent Crude (BRENT)": brent_df,
        "WTI Crude (WTI)": wti_df,
        "USD/JPY (USDJPY)": usdjpy_df
    }

    # -------------------------------------------------------------
    # 1. Baseline Model A & B Equity Signals
    # -------------------------------------------------------------
    print("\n[1] Testing Model A vs Model B Multi-Horizon Signals across Equities:")
    print("-" * 110)
    for eq_name, eq_df in equities.items():
        res_a = generate_macro_full_signals(eq_df, tnx_df, brent_df, wti_df, equity_name=eq_name, all_dfs=all_dfs)
        sig_a = res_a["signals"]
        res_b = generate_rate_only_signals(eq_df, tnx_df, equity_name=eq_name, all_dfs=all_dfs)
        sig_b = res_b["signals"]

        print(f"ASSET: {eq_name:<20} | Model A 10m: {sig_a['forward_10m']['action']:<10} (P={sig_a['forward_10m']['prob_up']*100:.1f}%) | Model B 10m: {sig_b['forward_10m']['action']:<10} (P={sig_b['forward_10m']['prob_up']*100:.1f}%)")

        assert "forward_10m" in sig_a, "Missing forward_10m in Model A"
        assert "action" in sig_a["forward_10m"], "Missing action Choice in Model A"
        assert "prob_up" in sig_a["forward_10m"], "Missing prob_up Noul in Model A"
        assert 0.0 <= sig_a["forward_10m"]["prob_up"] <= 1.0, "Invalid prob_up Noul range"

    # -------------------------------------------------------------
    # 2. Dynamic Nikkei 225 USD/JPY FX vs Equity Sentiment Engine
    # -------------------------------------------------------------
    print("\n[2] Testing Dynamic USD/JPY FX vs Equity Sentiment Conditioning for Nikkei 225:")
    print("-" * 110)

    # Scenario A: USD/JPY surging (+0.6% in 5m), SPX flat -> FX_DOMINANT regime
    usdjpy_surging = ni_df.copy()
    close_vals = np.linspace(155.0, 158.0, len(dates)) # Strong upward trend in USD/JPY (weakening yen)
    usdjpy_surging["Close"] = close_vals
    usdjpy_surging["High"] = close_vals + 0.1
    usdjpy_surging["Low"] = close_vals - 0.1
    usdjpy_surging["Open"] = close_vals - 0.05
    usdjpy_surging["Volume"] = 1000

    spx_flat = sp_df.copy()
    spx_flat["Close"] = 5800.0 + np.random.normal(0, 0.01, len(dates))

    dyn_a = compute_nikkei_dynamic_fx_equity_regime(ni_df, usdjpy_df=usdjpy_surging, spx_df=spx_flat, all_dfs=None)
    print("Scenario A (USD/JPY Surging +0.6%, SPX Flat):")
    print(f"  * Regime:             {dyn_a['regime']} (Choice)")
    print(f"  * USD/JPY FX Weight:  {dyn_a['fx_weight'] * 100:.1f}%")
    print(f"  * Equity Weight:      {dyn_a['equity_weight'] * 100:.1f}%")
    print(f"  * FX Sensitivity:     {dyn_a['fx_sensitivity_score']:.1f}/100 (Score)")
    print(f"  * FX Transmission:    {dyn_a['prob_fx_transmission'] * 100:.1f}% (Noul)")
    print(f"  * Dynamic Macro Adj:  {dyn_a['dynamic_macro_impact']:+.4f}")
    assert dyn_a["fx_weight"] > 0.50, f"Expected fx_weight > 0.50 when FX is surging, got {dyn_a['fx_weight']}"
    assert dyn_a["dynamic_macro_impact"] > 0.0, "Expected positive macro boost on Nikkei from weakening Yen"
    assert dyn_a["regime"] in ["FX_DOMINANT", "DUAL_REINFORCING"], f"Unexpected regime: {dyn_a['regime']}"

    # Scenario B: USD/JPY plunging (-0.6% in 5m, Yen surge / carry unwind), SPX flat -> Bearish Drag
    usdjpy_plunging = usdjpy_surging.copy()
    plunge_vals = np.linspace(158.0, 155.0, len(dates))
    usdjpy_plunging["Close"] = plunge_vals
    usdjpy_plunging["High"] = plunge_vals + 0.1
    usdjpy_plunging["Low"] = plunge_vals - 0.1

    dyn_b = compute_nikkei_dynamic_fx_equity_regime(ni_df, usdjpy_df=usdjpy_plunging, spx_df=spx_flat, all_dfs=None)
    print("\nScenario B (USD/JPY Plunging -0.6%, Yen Surge):")
    print(f"  * Regime:             {dyn_b['regime']} (Choice)")
    print(f"  * USD/JPY FX Weight:  {dyn_b['fx_weight'] * 100:.1f}%")
    print(f"  * Dynamic Macro Adj:  {dyn_b['dynamic_macro_impact']:+.4f}")
    assert dyn_b["dynamic_macro_impact"] < 0.0, "Expected negative macro drag on Nikkei from rapid Yen appreciation"

    # Scenario C: Opposing drivers (USD/JPY dropping, SPX surging) -> DECOUPLED_CHOP
    spx_surging = sp_df.copy()
    spx_vals = np.linspace(5750.0, 5850.0, len(dates))
    spx_surging["Close"] = spx_vals
    dyn_c = compute_nikkei_dynamic_fx_equity_regime(ni_df, usdjpy_df=usdjpy_plunging, spx_df=spx_surging, all_dfs=None)
    print("\nScenario C (USD/JPY Plunging vs SPX Surging):")
    print(f"  * Regime:             {dyn_c['regime']} (Choice)")
    print(f"  * Description:        {dyn_c['regime_desc']}")
    assert dyn_c["regime"] == "DECOUPLED_CHOP", f"Expected DECOUPLED_CHOP, got {dyn_c['regime']}"

    # -------------------------------------------------------------
    # 2B. Nikkei 225 USD/JPY FX vs Pure Equity Decomposition Engine
    # -------------------------------------------------------------
    print("\n[2B] Testing Econometric 2-Factor Decomposition & Multi-Horizon Attribution:")
    print("-" * 110)
    decomp = decompose_nikkei_movements(ni_df, usdjpy_df=usdjpy_surging, spx_df=spx_surging, all_dfs=None)
    assert decomp["has_decomposition"] is True, "Decomposition flagged as inactive"
    assert "betas" in decomp, "Missing betas dictionary"
    assert "variance_decomposition" in decomp, "Missing variance_decomposition dictionary"
    assert "horizons" in decomp, "Missing horizons attribution dictionary"

    betas = decomp["betas"]
    v_dec = decomp["variance_decomposition"]
    print(f"  * Rolling Factor Betas:   β_FX={betas['beta_fx']:+.4f}, β_SPX_ortho={betas['beta_equity_ortho']:+.4f}, γ_SPX_on_FX={betas['gamma_spx_fx']:+.4f}")
    print(f"  * ANOVA Variance Shares:  FX={v_dec['fx_var_pct']:.1f}%, Pure Equity={v_dec['pure_equity_var_pct']:.1f}%, Idiosyncratic={v_dec['idiosyncratic_var_pct']:.1f}% | R²={v_dec['r_squared_pct']:.1f}%")
    print(f"  * Primary Driver Choice:  {decomp['primary_driver']} ({decomp['primary_driver_desc']})")

    # Verify variance shares sum to ~100%
    var_total_pct = v_dec["fx_var_pct"] + v_dec["pure_equity_var_pct"] + v_dec["idiosyncratic_var_pct"]
    assert abs(var_total_pct - 100.0) < 1.0, f"Expected variance shares ~100%, got {var_total_pct}%"
    assert 0.0 <= v_dec["r_squared"] <= 1.0, f"Invalid R^2: {v_dec['r_squared']}"

    # Verify Exact Additive Horizon Identity: Delta_NK == Delta_FX + Delta_PureEquity + Delta_Idiosyncratic
    for h_name, h_vals in decomp["horizons"].items():
        delta_nk = h_vals["ret_nikkei_pct"]
        delta_sum = h_vals["fx_impact_pct"] + h_vals["pure_equity_impact_pct"] + h_vals["idiosyncratic_impact_pct"]
        assert abs(delta_nk - delta_sum) < 1e-4, f"Additive identity violated for {h_name}: NK={delta_nk} != sum={delta_sum}"
        print(f"  * Horizon {h_name:>3} Attribution:  Nikkei={delta_nk:+.3f}% (FX={h_vals['fx_impact_pct']:+.3f}%, Eq={h_vals['pure_equity_impact_pct']:+.3f}%, Idio={h_vals['idiosyncratic_impact_pct']:+.3f}%) [Shares: {h_vals['share_fx_pct']:.0f}% FX, {h_vals['share_equity_pct']:.0f}% Eq]")

    # Verify Quanto / Exporter Margin dynamics
    q_dyn = decomp.get("quanto_dynamics", {})
    assert "usd_jpy_rate" in q_dyn, "Missing USD/JPY rate in quanto dynamics"
    assert "exporter_margin_spread_5m_bps" in q_dyn, "Missing exporter margin spread"
    print(f"  * Quanto / Exporter Spread: USD/JPY={q_dyn['usd_jpy_rate']}, Exporter Margin={q_dyn['exporter_margin_spread_5m_bps']:+.1f} bps")

    # Synthetic Known Factor Loadings Recovery Test (Beta_FX = 0.85, Beta_SPX = 0.65)
    np.random.seed(123)
    dates_synth = pd.date_range(end=pd.Timestamp.now(), periods=50, freq="1min")
    fx_rets_known = np.random.normal(0.0005, 0.002, 50)
    sp_rets_known = np.random.normal(0.0003, 0.003, 50)
    noise_known = np.random.normal(0.0, 0.0004, 50)
    ni_rets_known = 0.85 * fx_rets_known + 0.65 * sp_rets_known + noise_known

    fx_series = 155.0 * np.cumprod(1.0 + fx_rets_known)
    sp_series = 5800.0 * np.cumprod(1.0 + sp_rets_known)
    ni_series = 38000.0 * np.cumprod(1.0 + ni_rets_known)

    df_fx = pd.DataFrame({"Close": fx_series, "High": fx_series+0.1, "Low": fx_series-0.1, "Open": fx_series, "Volume": 1000}, index=dates_synth)
    df_sp = pd.DataFrame({"Close": sp_series, "High": sp_series+1.0, "Low": sp_series-1.0, "Open": sp_series, "Volume": 1000}, index=dates_synth)
    df_ni = pd.DataFrame({"Close": ni_series, "High": ni_series+10.0, "Low": ni_series-10.0, "Open": ni_series, "Volume": 1000}, index=dates_synth)

    decomp_synth = decompose_nikkei_movements(df_ni, usdjpy_df=df_fx, spx_df=df_sp)
    print(f"  * Ground Truth Recovery:  β_FX={decomp_synth['betas']['beta_fx']:.2f} (Target 0.85), β_SPX_raw={decomp_synth['betas']['beta_equity_raw']:.2f} (Target 0.65), R²={decomp_synth['variance_decomposition']['r_squared_pct']:.1f}%")
    assert decomp_synth['variance_decomposition']['r_squared'] > 0.60, f"Expected R^2 > 0.60, got {decomp_synth['variance_decomposition']['r_squared']}"
    assert decomp_synth['betas']['beta_fx'] > 0.50, f"Expected Beta_FX > 0.50, got {decomp_synth['betas']['beta_fx']}"
    assert decomp_synth['betas']['beta_equity_raw'] > 0.40, f"Expected Beta_SPX > 0.40, got {decomp_synth['betas']['beta_equity_raw']}"

    # -------------------------------------------------------------
    # 3. TypeSafe Jev System One Output Variables Specification
    # -------------------------------------------------------------
    print("\n[3] Verifying TypeSafe Jev System One Output Primitives (Score, Choice, Noul):")
    print("-" * 110)

    # 1. SCORE Primitive: Continuous calibrated metric with confidence
    print("Primitive 1: SCORE")
    score_info = JEV_PRIMITIVES_GUIDE["Score"]
    print(f"  * Definition: {score_info['definition'][:120]}...")
    print(f"  * Referenced Variables: {list(score_info['forecasted_variables_referenced'].keys())}")
    res_nikkei = generate_macro_full_signals(ni_df, tnx_df, brent_df, wti_df, equity_name="Nikkei 225 (NI225)", all_dfs=all_dfs)
    conviction = res_nikkei.get("conviction_score")
    assert conviction is not None, "Missing conviction_score"
    score_val = conviction.get("score") if isinstance(conviction, dict) else float(conviction)
    print(f"  * Sample Nikkei Conviction Score: {score_val} (Scale: -100 to +100)")
    assert -100.0 <= score_val <= 100.0, "Score outside range [-100, 100]"

    # 2. CHOICE Primitive: Discrete categorical selection with probabilities
    print("\nPrimitive 2: CHOICE")
    choice_info = JEV_PRIMITIVES_GUIDE["Choice"]
    print(f"  * Definition: {choice_info['definition'][:120]}...")
    print(f"  * Referenced Variables: {list(choice_info['forecasted_variables_referenced'].keys())}")
    tactical_choice = res_nikkei["signals"]["forward_10m"]["action"]
    print(f"  * Sample Nikkei 10m Action Choice: {tactical_choice}")
    assert tactical_choice in ["STRONG_LONG", "LEAN_LONG", "NEUTRAL", "LEAN_SHORT", "STRONG_SHORT", "BUY_LONG", "HOLD_CASH", "SELL_SHORT"], f"Invalid Choice: {tactical_choice}"
    dyn_regime = res_nikkei.get("nikkei_fx_equity_dynamics", {}).get("regime")
    print(f"  * Sample Nikkei Driver Regime Choice: {dyn_regime}")
    assert dyn_regime in ["FX_DOMINANT", "EQUITY_SENTIMENT_DOMINANT", "DUAL_REINFORCING", "DECOUPLED_CHOP"]

    # 3. NOUL Primitive: Calibrated truth probability without separate confidence
    print("\nPrimitive 3: NOUL")
    noul_info = JEV_PRIMITIVES_GUIDE["Noul"]
    print(f"  * Definition: {noul_info['definition'][:120]}...")
    print(f"  * Referenced Variables: {list(noul_info['forecasted_variables_referenced'].keys())}")
    prob_10m = res_nikkei["signals"]["forward_10m"]["prob_up"]
    prob_trans = res_nikkei.get("nikkei_fx_equity_dynamics", {}).get("prob_fx_transmission")
    print(f"  * Sample 10m Prob Up Noul:        {prob_10m * 100:.2f}%")
    print(f"  * Sample Prob FX Transmission:     {prob_trans * 100:.2f}%")
    assert 0.0 <= prob_10m <= 1.0, "prob_10m Noul out of bounds"
    assert 0.0 <= prob_trans <= 1.0, "prob_trans Noul out of bounds"
    print("  * Verification: Noul directly embeds uncertainty via P (P=0.50 is maximum entropy).")

    # -------------------------------------------------------------
    # 4. Institutional OANDA Feed Default Priority
    # -------------------------------------------------------------
    print("\n[4] Verifying Institutional OANDA CFD Feed Configuration:")
    print("-" * 110)
    from oanda_feed import OANDA_INSTRUMENTS, get_configured_credentials, test_oanda_connection
    key, env = get_configured_credentials()
    print(f"  * OANDA Configured: Key Present = {bool(key)}, Environment = {env.upper()}")
    print(f"  * Instruments Supported ({len(OANDA_INSTRUMENTS)} Total):")
    for name, oanda_code in OANDA_INSTRUMENTS.items():
        print(f"      - {name:<22} -> {oanda_code}")
    assert "USD/JPY (USDJPY)" in OANDA_INSTRUMENTS, "USD/JPY missing from OANDA_INSTRUMENTS"
    assert OANDA_INSTRUMENTS["USD/JPY (USDJPY)"] == "USD_JPY", "Incorrect OANDA instrument mapping"
    assert "Nikkei 225 (NI225)" in OANDA_INSTRUMENTS, "Nikkei 225 missing from OANDA_INSTRUMENTS"

    print("\n" + "=" * 110)
    print("     ALL MACRO CONDITIONING, NIKKEI FX/EQUITY & JEV PRIMITIVES TESTS PASSED 100%!")
    print("=" * 110)

if __name__ == "__main__":
    run_tests()
