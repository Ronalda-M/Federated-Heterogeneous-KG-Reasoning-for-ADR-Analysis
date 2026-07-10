"""Tests for evaluation metrics."""
import pytest
import numpy as np
import torch
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_ranking_metrics,
    evaluate_subgroup,
    TemperatureScaler,
)

def _make_data(n=200, pos_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_ratio).astype(int)
    scores = np.where(y, rng.random(n) * 0.5 + 0.5,
                          rng.random(n) * 0.5)
    return y, scores

def test_classification_metrics_keys():
    y, s = _make_data()
    m = compute_classification_metrics(y, s)
    for k in ["precision","recall","f1","roc_auc","pr_auc","threshold"]:
        assert k in m

def test_roc_auc_range():
    y, s = _make_data()
    m = compute_classification_metrics(y, s)
    assert 0.0 <= m["roc_auc"] <= 1.0

def test_ranking_metrics():
    y, s = _make_data(n=100, pos_ratio=0.2)
    m = compute_ranking_metrics(s, y, k_values=[1, 5, 10])
    assert "mrr" in m
    assert "hits@1" in m and "hits@5" in m and "hits@10" in m
    assert 0.0 <= m["mrr"] <= 1.0

def test_subgroup_evaluation():
    y, s = _make_data(n=300, pos_ratio=0.3)
    groups = np.array([0]*100 + [1]*100 + [2]*100)
    labels = ["male","female","unknown"]
    res = evaluate_subgroup(y, s, groups, labels)
    assert "male" in res
    assert "pr_auc" in res["male"]

def test_temperature_scaler():
    scaler = TemperatureScaler()
    logits = torch.randn(100)
    labels = (torch.rand(100) > 0.5).float()
    T = scaler.fit(logits, labels)
    assert T > 0
    probs = scaler(logits)
    assert (probs >= 0).all() and (probs <= 1).all()
