"""
Validation Suite for Real-Time Shipping Telemetry & Brent Crude Forward Estimation
Fuses Oil_Tanker_Traffic_AntiGravity AIS data with TypeSafe Jev Multi-Horizon System.
"""

import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import json
import numpy as np
import pandas as pd
from maritime_brent_momentum import (
    generate_maritime_brent_signals,
    get_live_shipping_telemetry,
    compute_maritime_physical_supply_index
)

def run_maritime_brent_test():
    print("=" * 95)
    print("   MARITIME AIS TANKER TRAFFIC -> BRENT CRUDE FORWARD ESTIMATION VERIFICATION")
    print("=" * 95)

    # 1. Inspect live shipping telemetry
    telemetry = get_live_shipping_telemetry()
    source = telemetry.get("source", "unknown")
    print(f"\n[1] Shipping Telemetry Source: {source}")
    h = telemetry.get("hormuz", {})
    r = telemetry.get("redsea_bab_el_mandeb", {})
    e = telemetry.get("energy_state", {})

    print(f"    - Strait of Hormuz: {h.get('active_vessels_counted')} active tankers | Clandestine Rate: {h.get('clandestine_rate_pct'):.1f}%")
    print(f"    - Bab El-Mandeb / Red Sea: {r.get('traffic_today')} daily vs {r.get('historical_5yr_avg')} 5yr avg | Cape Rerouting Delay: +{r.get('cape_diversion_delay_days')} days")
    print(f"    - Singapore Gasoil Crack: ${e.get('singapore_gasoil_crack_usd'):.2f}/bbl | Floating Storage: {e.get('fujairah_asean_floating_storage_mbbls')}M bbls")
    print(f"    - Forward Curve: {e.get('forward_curve_structure')} (Spread: {e.get('prompt_to_m6_spread')})")

    # 2. Compute Maritime Physical Supply Index (MPSI)
    mpsi = compute_maritime_physical_supply_index(telemetry)
    print(f"\n[2] Maritime Physical Supply Index (MPSI):")
    print(f"    - Z-Score: {mpsi['z_maritime']:+.3f}σ")
    print(f"    - Regime Classification: {mpsi['regime']}")
    print(f"    - Directional Physical Bias: {mpsi['bias']}")
    print(f"    - Components: {mpsi['components']}")

    # 3. Create simulated 1-minute Brent Crude candles
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=75, freq="1min")
    prices = 104.50 + np.cumsum(np.random.normal(0.015, 0.06, len(dates)))
    brent_df = pd.DataFrame({
        "Open": prices - 0.04,
        "High": prices + 0.10,
        "Low": prices - 0.08,
        "Close": prices,
        "Volume": np.random.randint(100, 500, len(dates))
    }, index=dates)

    # 4. Generate forward estimation signals
    print(f"\n[3] Generating Multi-Horizon Forward Trend Signals (TypeSafe Jev + Wave Oscillator)...")
    res = generate_maritime_brent_signals(brent_df)
    sigs = res["signals"]

    print(f"    - Engine Status: {sigs.get('status')}")
    print(f"    - Alignment: {sigs.get('alignment')}")
    print(f"    - Average Prob(Up): {sigs.get('average_prob_up')*100:.1f}%")

    proj = sigs.get("forward_projections", {})
    curr = proj.get("current_price", brent_df['Close'].iloc[-1])
    print(f"\n[4] Quantitative Forward Price Estimates (Base Prompt: ${curr:.2f}):")
    print(f"    - 1-Minute Forward  (t+1m) : ${proj.get('pred_1m'):.2f} [{sigs['forward_1m']['action']} - P(Up)={sigs['forward_1m']['prob_up']*100:.1f}%, Conf={sigs['forward_1m']['confidence']*100:.0f}%]")
    print(f"    - 10-Minute Forward (t+10m): ${proj.get('pred_10m'):.2f} [{sigs['forward_10m']['action']} - P(Up)={sigs['forward_10m']['prob_up']*100:.1f}%, Conf={sigs['forward_10m']['confidence']*100:.0f}%]")
    print(f"    - 30-Minute Forward (t+30m): ${proj.get('pred_30m'):.2f} [{sigs['forward_30m']['action']} - P(Up)={sigs['forward_30m']['prob_up']*100:.1f}%, Conf={sigs['forward_30m']['confidence']*100:.0f}%]")
    print(f"    - 1-Hour Forward    (t+60m): ${proj.get('pred_1h'):.2f} [{sigs['forward_1h']['action']} - P(Up)={sigs['forward_1h']['prob_up']*100:.1f}%, Conf={sigs['forward_1h']['confidence']*100:.0f}%]")

    assert sigs["forward_1m"]["prob_up"] is not None
    assert sigs["forward_10m"]["prob_up"] is not None
    assert sigs["forward_30m"]["prob_up"] is not None
    assert sigs["forward_1h"]["prob_up"] is not None

    # 5. Verify AIS Efficacy Confusion Matrix
    print(f"\n[5] AIS Predictive Efficacy Confusion Matrix Analysis:")
    assert "ais_confusion_matrix" in res, "ais_confusion_matrix missing from signals output!"
    ais_cm = res["ais_confusion_matrix"]
    cm_sum = ais_cm["summary"]
    cm_horiz = ais_cm["horizons"]

    print(f"    - Baseline Overall Hit Rate: {cm_sum['overall_baseline_hit_rate']:.1f}%")
    print(f"    - AIS-Conditioned Hit Rate : {cm_sum['overall_ais_hit_rate']:.1f}%")
    print(f"    - Alpha Lift               : {cm_sum['overall_hit_rate_lift_pct']:+.1f}%")
    print(f"    - False Breakdowns Rescued : {cm_sum['total_whipsaws_rescued']} shorts vetoed")
    print(f"    - Supply Breakouts Added   : {cm_sum['total_breakouts_confirmed']} longs confirmed")
    print(f"    - Empirical Verdict        : {cm_sum['verdict']}")

    for h_key in ["1m", "10m", "30m", "1h"]:
        h = cm_horiz[h_key]
        print(f"      * {h['label']:16s}: Base={h['baseline']['hit_rate']:4.1f}% | AIS={h['ais_conditioned']['hit_rate']:4.1f}% | Lift={h['delta_hit_rate']:+5.1f}% | Rescued={h['whipsaws_rescued']:2d} | Status={h['status']}")

    assert cm_sum["overall_hit_rate_lift_pct"] is not None
    assert len(cm_horiz) == 4

    print("\n[SUCCESS] All forward estimation and AIS confusion matrix tests passed successfully!")

if __name__ == "__main__":
    run_maritime_brent_test()
