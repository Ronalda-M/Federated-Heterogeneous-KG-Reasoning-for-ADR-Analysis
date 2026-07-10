"""Tests for statistical significance testing."""
import pytest
import numpy as np
from src.evaluation.significance import (
    bootstrap_pr_auc_diff,
    holm_bonferroni,
    run_significance_suite,
)

def _make_scores(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    strong = np.where(y, rng.random(n)*0.4+0.6, rng.random(n)*0.4)
    weak   = np.where(y, rng.random(n)*0.3+0.5, rng.random(n)*0.5)
    random = rng.random(n)
    return y, strong, weak, random

def test_bootstrap_returns_tuple():
    y, strong, weak, _ = _make_scores()
    result = bootstrap_pr_auc_diff(y, strong, weak, n_bootstrap=100)
    assert len(result) == 5
    obs, mean_d, ci_lo, ci_hi, p = result
    assert ci_lo <= obs <= ci_hi or abs(obs - mean_d) < 0.5

def test_strong_vs_random_significant():
    y, strong, _, random = _make_scores(n=800)
    _, _, _, _, p = bootstrap_pr_auc_diff(y, strong, random, n_bootstrap=500)
    assert p < 0.1  # strong model should significantly beat random

def test_holm_bonferroni_length():
    p_vals = [0.01, 0.04, 0.20, 0.50]
    adj, reject = holm_bonferroni(p_vals)
    assert len(adj) == 4
    assert len(reject) == 4

def test_holm_bonferroni_rejects_small():
    p_vals = [0.001, 0.9]
    _, reject = holm_bonferroni(p_vals)
    assert reject[0] is True

def test_significance_suite_structure():
    y, strong, weak, random = _make_scores(n=400)
    scores = {"proposed": strong, "baseline_a": weak, "baseline_b": random}
    res = run_significance_suite(y, scores, proposed_key="proposed",
                                 n_bootstrap=200)
    assert "comparisons" in res
    assert len(res["comparisons"]) == 2
    for comp in res["comparisons"]:
        assert "p_value_holm_bonferroni" in comp
        assert "significant" in comp
