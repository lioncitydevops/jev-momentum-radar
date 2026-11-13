"""
Comprehensive Unit & Integration Test Suite for Bayesian Mathematical Forecasting
Validates TypeSafe Jev System One Primitives (Choice, Score, Noul) coupled with
Conjugate Beta-Binomial Distributions, Log-Odds Information Updating, Harold Jeffreys Evidence Scale,
Continuous Predictive Price Envelopes, and Shannon Epistemic Uncertainty.
"""

import sys
import math
import numpy as np
import pytest

from bayesian_forecasting import (
    logit,
    sigmoid,
    classify_bayes_factor_jeffreys,
    compute_beta_credible_interval,
    compute_epistemic_entropy_bits,
    compute_bayesian_horizon_forecast,
    compute_multi_horizon_bayesian_forecast
)

def test_logit_sigmoid_inversion():
    """Verify logit and sigmoid are mathematical inverses across (0, 1)."""
    test_probs = [0.01, 0.05, 0.20, 0.50, 0.75, 0.90, 0.99]
    for p in test_probs:
        lo = logit(p)
        p_recovered = sigmoid(lo)
        assert abs(p - p_recovered) < 1e-6, f"Inversion failed for p={p}: got {p_recovered}"

def test_jeffreys_scale_classification():
    """Verify exact categorization under Sir Harold Jeffreys' scientific evidence scale."""
    cat, badge, _ = classify_bayes_factor_jeffreys(120.0)
    assert cat == "DECISIVE_BULLISH" and "Decisive" in badge

    cat, badge, _ = classify_bayes_factor_jeffreys(15.0)
    assert cat == "STRONG_BULLISH" and "Strong" in badge

    cat, badge, _ = classify_bayes_factor_jeffreys(5.5)
    assert cat == "SUBSTANTIAL_BULLISH" and "Substantial" in badge

    cat, badge, _ = classify_bayes_factor_jeffreys(2.0)
    assert cat == "ANECDOTAL_BULLISH" and "Anecdotal" in badge

    cat, badge, _ = classify_bayes_factor_jeffreys(1.0)
    assert cat == "INCONCLUSIVE" and "Inconclusive" in badge

    cat, badge, _ = classify_bayes_factor_jeffreys(0.80)
    assert cat == "ANECDOTAL_BEARISH" and "Anecdotal" in badge

    cat, badge, _ = classify_bayes_factor_jeffreys(0.25)
    assert cat == "SUBSTANTIAL_BEARISH" and "Substantial" in badge

    cat, badge, _ = classify_bayes_factor_jeffreys(0.005)
    assert cat == "DECISIVE_BEARISH" and "Decisive" in badge

def test_beta_credible_interval():
    """Verify conjugate Beta-Binomial 95% credible intervals."""
    # Balanced prior
    low, high = compute_beta_credible_interval(10.0, 10.0, level=0.95)
    assert 0.20 < low < 0.35, f"Unexpected low={low}"
    assert 0.65 < high < 0.80, f"Unexpected high={high}"
    assert abs((low + high) / 2.0 - 0.50) < 0.05, "Symmetric Beta(10,10) should center near 0.50"

    # Bullish posterior
    b_low, b_high = compute_beta_credible_interval(25.0, 5.0, level=0.95)
    assert b_low > 0.65, f"Bullish low should be high, got {b_low}"
    assert b_high <= 1.0

def test_epistemic_entropy():
    """Verify Shannon entropy reaches maximum 1.0 bit at p=0.50 and approaches 0 at extremes."""
    assert abs(compute_epistemic_entropy_bits(0.50) - 1.0) < 1e-4
    assert compute_epistemic_entropy_bits(0.99) < 0.15
    assert compute_epistemic_entropy_bits(0.01) < 0.15
    assert compute_epistemic_entropy_bits(0.75) < 1.0

def test_bayesian_horizon_forecast_monotonicity():
    """
    Verify that when Jev indicates high bullish probability,
    the posterior probability shifts higher than the prior,
    yielding a Bayes Factor > 1.0.
    """
    prior_prob = 0.52
    jev_prob = 0.70
    jev_conf = 0.85
    curr_price = 5000.0
    features = {"price": curr_price, "atr_14": 15.0, "vol_gk": 0.001}

    res = compute_bayesian_horizon_forecast(
        horizon_name="10m",
        prior_prob=prior_prob,
        jev_noul=jev_prob,
        jev_choice="STRONG_LONG",
        jev_confidence=jev_conf,
        features=features,
        tau_minutes=10
    )

    assert res["prob_up"] > prior_prob, f"Posterior {res['prob_up']} should be > prior {prior_prob}"
    assert res["bayes_factor"] > 1.0, f"Bayes Factor {res['bayes_factor']} should be > 1.0"
    assert "Bullish" in res["evidence_badge"]
    assert res["action"] in ["BUY", "STRONG_BUY", "STRONG_LONG", "LEAN_LONG"]
    
    # Check continuous predictive price envelope
    cp = res["continuous_predictive"]
    assert cp["pred_forward_price"] > curr_price
    assert cp["credible_price_interval_95"]["price_lower_95"] < cp["pred_forward_price"]
    assert cp["credible_price_interval_95"]["price_upper_95"] > cp["pred_forward_price"]
    assert cp["indicative_protective_stop"] < curr_price

def test_multi_horizon_bayesian_forecast():
    """Verify complete multi-horizon engine processing with mock Jev answers."""
    prior_dict = {
        "forward_1m": {"action": "BUY", "prob_up": 0.55, "confidence": 0.65},
        "forward_10m": {"action": "BUY", "prob_up": 0.58, "confidence": 0.70},
        "forward_30m": {"action": "BUY", "prob_up": 0.60, "confidence": 0.72},
        "forward_1h": {"action": "BUY", "prob_up": 0.62, "confidence": 0.75}
    }
    jev_answers = {
        "signal_1m_action": {"choice": "BUY", "confidence": 0.80},
        "prob_up_1m": {"noul": 0.65},
        "signal_10m_action": {"choice": "STRONG_BUY", "confidence": 0.88},
        "prob_up_10m": {"noul": 0.72},
        "signal_30m_action": {"choice": "STRONG_BUY", "confidence": 0.85},
        "prob_up_30m": {"noul": 0.70},
        "signal_1h_action": {"choice": "BUY", "confidence": 0.82},
        "prob_up_1h": {"noul": 0.68},
        "momentum_conviction_score": {"score": 3.4}
    }
    eq_feat = {"price": 20000.0, "atr_14": 40.0, "vol_gk": 0.0012}

    mh_res = compute_multi_horizon_bayesian_forecast(prior_dict, jev_answers, eq_feat)

    assert "horizons" in mh_res
    assert len(mh_res["horizons"]) == 4
    for h in ["1m", "10m", "30m", "1h"]:
        assert h in mh_res["horizons"]
        h_obj = mh_res["horizons"][h]
        assert "bayes_factor" in h_obj
        assert "evidence_badge" in h_obj
        assert "beta_posterior" in h_obj
        assert "continuous_predictive" in h_obj
        assert h_obj["prob_up"] > 0.50

    assert "forward_projections" in mh_res
    assert mh_res["forward_projections"]["pred_10m"] > 20000.0
    assert "STRONG_BULLISH_ALIGNMENT" in mh_res["alignment"]
    assert mh_res["multi_horizon_geometric_mean_bf"] > 1.0

def test_horizon_comparative_advantage_scaling():
    """
    Verify that for identical Jev evidence, the 1m horizon produces a much smaller
    Weight of Evidence (gamma=0.35) than the 1h horizon (gamma=1.50), mathematically
    preventing LLM semantic signals from overriding 1m high-frequency tape physics.
    """
    feat = {"price": 100.0, "atr_14": 0.5, "vol_gk": 0.001}
    res_1m = compute_bayesian_horizon_forecast("1m", 0.50, 0.70, "BUY", 0.80, feat, tau_minutes=1)
    res_1h = compute_bayesian_horizon_forecast("1h", 0.50, 0.70, "BUY", 0.80, feat, tau_minutes=60)

    assert res_1m["bayes_factor"] < res_1h["bayes_factor"], "1m BF must be strictly smaller than 1h BF under comparative advantage"
    assert res_1m["posterior_prob_up"] < res_1h["posterior_prob_up"], "1m posterior should be more conservative than 1h posterior"
    assert res_1m["uncertainty_decomposition"]["horizon_semantic_weight_gamma"] == 0.35
    assert res_1h["uncertainty_decomposition"]["horizon_semantic_weight_gamma"] == 1.50

def test_closed_loop_confusion_matrix_shrinkage():
    """
    Verify that when walk-forward confusion matrix edge is negative (model underperforming),
    omega_edge collapses, safely reverting towards the objective prior.
    """
    feat = {"price": 100.0, "atr_14": 0.5, "vol_gk": 0.001}
    opt_positive = {"confusion_matrix": {"10m": {"edge": 10.0}}}
    opt_negative = {"confusion_matrix": {"10m": {"edge": -5.0}}}

    res_pos = compute_bayesian_horizon_forecast("10m", 0.50, 0.70, "BUY", 0.80, feat, tau_minutes=10, opt_configs=opt_positive)
    res_neg = compute_bayesian_horizon_forecast("10m", 0.50, 0.70, "BUY", 0.80, feat, tau_minutes=10, opt_configs=opt_negative)

    assert res_pos["uncertainty_decomposition"]["walk_forward_shrinkage_omega"] > res_neg["uncertainty_decomposition"]["walk_forward_shrinkage_omega"]
    assert res_pos["bayes_factor"] > res_neg["bayes_factor"]

def test_information_theoretic_ess_at_maximum_entropy():
    """
    Verify that when Jev is in maximum uncertainty (p_Jev = 0.50, H = 1.0 bit),
    pseudo-counts n_Jev approach zero, meaning Jev adds almost zero observations to Beta posterior.
    """
    feat = {"price": 100.0, "atr_14": 0.5, "vol_gk": 0.001}
    res_entropy = compute_bayesian_horizon_forecast("10m", 0.50, 0.50, "NEUTRAL", 0.50, feat, tau_minutes=10)
    beta_post = res_entropy["beta_posterior"]
    
    # Prior ESS is around 14. When Jev is neutral, total ESS should be close to prior ESS
    assert abs(beta_post["mean"] - 0.50) < 1e-4
    assert res_entropy["bayes_factor"] == 1.0
    assert "Inconclusive" in res_entropy["evidence_badge"]

if __name__ == "__main__":
    print("Running Bayesian Forecasting Test Suite...")
    test_logit_sigmoid_inversion()
    print("✓ Logit & Sigmoid Inversion passed.")
    test_jeffreys_scale_classification()
    print("✓ Harold Jeffreys Scale Classification passed.")
    test_beta_credible_interval()
    print("✓ Beta-Binomial 95% Credible Interval passed.")
    test_epistemic_entropy()
    print("✓ Epistemic Shannon Entropy passed.")
    test_bayesian_horizon_forecast_monotonicity()
    print("✓ Bayesian Horizon Forecast Monotonicity passed.")
    test_multi_horizon_bayesian_forecast()
    print("✓ Multi-Horizon Bayesian Forecast Pipeline passed.")
    test_horizon_comparative_advantage_scaling()
    print("✓ Horizon Comparative Advantage Precision Scaling gamma(tau) passed.")
    test_closed_loop_confusion_matrix_shrinkage()
    print("✓ Closed-Loop Walk-Forward Shrinkage omega_edge passed.")
    test_information_theoretic_ess_at_maximum_entropy()
    print("✓ Information-Theoretic Effective Sample Size (ESS) at Maximum Entropy passed.")
    print("\nALL 9 THEORETICAL BAYESIAN TEST CASES PASSED WITH 100% SUCCESS!")
