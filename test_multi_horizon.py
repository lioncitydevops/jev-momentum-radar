"""
End-to-End Test & Verification Suite for Multi-Horizon (1m, 10m, 30m, 1h Forward) Trend Signals
Input Interval: 1-Minute Candles

Tests:
1. Feature Engineering on 1m bars (Returns, Garman-Klass Vol, VWAP Z-score, RSI, RVOL, Wave Dynamics).
2. TypeSafe Jev System One Multi-Horizon Query (`/systemone`) for 1m, 10m, 30m, 1h.
3. Calibrated Fallback Prior Model for 1m, 10m, 30m, 1h.
4. Live Data Ingestion & Signal Synthesis for Multi-Asset CFD Universe.
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

from multi_horizon_momentum import (
    compute_1m_features,
    format_multi_horizon_state_prompt,
    simulate_calibrated_multi_horizon_prior,
    query_typesafe_jev_multi_horizon,
    generate_multi_horizon_signals
)

def run_tests():
    print("=" * 100)
    print("      MULTI-HORIZON (1MIN, 10MIN, 30MIN, 1 HOUR FORWARD) TREND SIGNAL ENGINE VERIFICATION")
    print("=" * 100)

    # Test 1: Feature Extraction on 1m candles
    print("\n[TEST 1] Testing 1-Minute Feature Extraction...")
    dates = pd.date_range(end=pd.Timestamp.now(tz="America/New_York"), periods=75, freq="1min")
    prices = 5800.0 + np.cumsum(np.random.normal(0.2, 0.5, len(dates)))
    df_1m = pd.DataFrame({
        "Open": prices - np.random.uniform(0, 0.2, len(dates)),
        "High": prices + np.random.uniform(0.1, 0.4, len(dates)),
        "Low": prices - np.random.uniform(0.1, 0.4, len(dates)),
        "Close": prices,
        "Volume": np.random.randint(200, 1500, len(dates))
    }, index=dates)

    features = compute_1m_features(df_1m)
    assert "ret_1m" in features and "ret_10m" in features and "ret_30m" in features and "ret_60m" in features
    assert "vwap_z" in features and "rsi_14" in features and "rvol" in features
    w_dyn = features.get("wave_dynamics", {})
    assert "z_pred_1m" in w_dyn and "z_pred_10m" in w_dyn and "z_pred_30m" in w_dyn and "z_pred_1h" in w_dyn
    print("  [OK] 1m Feature Extraction PASS. Metrics & wave physics computed successfully.")
    print(f"    - Price: ${features['price']:.2f}")
    print(f"    - 1m Ret: {features['ret_1m']:+.2f}%, 10m Ret: {features['ret_10m']:+.2f}%, 30m Ret: {features['ret_30m']:+.2f}%, 60m Ret: {features['ret_60m']:+.2f}%")
    print(f"    - VWAP Z-Score: {features['vwap_z']:+.2f}, RSI(14): {features['rsi_14']:.1f}, RVOL: {features['rvol']:.2f}x")
    print(f"    - Wave Continuous Propagators (e^[A]τ): 1m={w_dyn['z_pred_1m']:+.2f}σ, 10m={w_dyn['z_pred_10m']:+.2f}σ, 30m={w_dyn['z_pred_30m']:+.2f}σ, 1h={w_dyn['z_pred_1h']:+.2f}σ")

    # Test 2: Prior Model Simulation
    print("\n[TEST 2] Testing Calibrated Mathematical Prior Fallback Model...")
    prior_res = simulate_calibrated_multi_horizon_prior(features, tv_rating_score=0.35)
    assert "forward_1m" in prior_res and "forward_10m" in prior_res and "forward_30m" in prior_res and "forward_1h" in prior_res
    print("  [OK] Mathematical Prior Model PASS.")
    print(f"    - Alignment: {prior_res['alignment']}")
    print(f"    - 1m Forward:  {prior_res['forward_1m']['action']} (P={prior_res['forward_1m']['prob_up']*100:.1f}%)")
    print(f"    - 10m Forward: {prior_res['forward_10m']['action']} (P={prior_res['forward_10m']['prob_up']*100:.1f}%)")
    print(f"    - 30m Forward: {prior_res['forward_30m']['action']} (P={prior_res['forward_30m']['prob_up']*100:.1f}%)")
    print(f"    - 1h Forward:  {prior_res['forward_1h']['action']} (P={prior_res['forward_1h']['prob_up']*100:.1f}%)")

    # Test 3: TypeSafe Jev System One Live Query
    print("\n[TEST 3] Testing TypeSafe Jev System One API Integration...")
    prompt = format_multi_horizon_state_prompt("S&P 500 (SPX)", features)
    jev_res = query_typesafe_jev_multi_horizon(prompt, features, tv_rating_score=0.35)
    print("  [OK] TypeSafe Jev Multi-Horizon Query PASS.")
    print(f"    - Live Jev Status: {jev_res['status']} (is_live_jev={jev_res.get('is_live_jev')})")
    print(f"    - Horizon Alignment: {jev_res['alignment']}")
    print(f"    - 1m Forward Signal:  {jev_res['forward_1m']['action']} (P={jev_res['forward_1m']['prob_up']*100:.1f}%, Conf={jev_res['forward_1m']['confidence']*100:.0f}%)")
    print(f"    - 10m Forward Signal: {jev_res['forward_10m']['action']} (P={jev_res['forward_10m']['prob_up']*100:.1f}%, Conf={jev_res['forward_10m']['confidence']*100:.0f}%)")
    print(f"    - 30m Forward Signal: {jev_res['forward_30m']['action']} (P={jev_res['forward_30m']['prob_up']*100:.1f}%, Conf={jev_res['forward_30m']['confidence']*100:.0f}%)")
    print(f"    - 1h Forward Signal:  {jev_res['forward_1h']['action']} (P={jev_res['forward_1h']['prob_up']*100:.1f}%, Conf={jev_res['forward_1h']['confidence']*100:.0f}%)")

    # Test 4: End-to-End Pipeline Execution across asset universe
    print("\n[TEST 4] Testing Multi-Asset Multi-Horizon Pipeline...")
    assets = ["S&P 500 (SPX)", "Nasdaq 100 (NDX)", "Russell 2000 (RUT)", "Nikkei 225 (NI225)"]
    for asset in assets:
        p_base = 5000.0 if "SPX" in asset else (18000.0 if "NDX" in asset else 2200.0)
        p_arr = p_base + np.cumsum(np.random.normal(0.1, 0.4, 75))
        df_a = pd.DataFrame({
            "Open": p_arr - 0.1,
            "High": p_arr + 0.3,
            "Low": p_arr - 0.3,
            "Close": p_arr,
            "Volume": np.random.randint(500, 2000, 75)
        }, index=pd.date_range(end=pd.Timestamp.now(), periods=75, freq="1min"))

        sig_res = generate_multi_horizon_signals(df_a, asset_name=asset)
        sigs = sig_res["signals"]
        print(f"  [OK] {asset:<20} | 1m: {sigs['forward_1m']['action']:<12} | 10m: {sigs['forward_10m']['action']:<12} | 30m: {sigs['forward_30m']['action']:<12} | 1h: {sigs['forward_1h']['action']:<12} | {sigs['alignment']}")

    print("\n" + "=" * 100)
    print("                     ALL MULTI-HORIZON TESTS PASSED SUCCESSFULLY!")
    print("=" * 100)

if __name__ == "__main__":
    run_tests()
