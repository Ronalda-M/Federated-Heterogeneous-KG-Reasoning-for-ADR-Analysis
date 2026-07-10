"""
Evaluation metrics for typed Drug--ADR link prediction.

Reported metrics:
    - Precision, Recall, F1 (at optimal threshold on val set)
    - ROC-AUC
    - PR-AUC
    - Mean Reciprocal Rank (MRR)
    - Hits@K  (K in {1, 5, 10})

Post-hoc calibration:
    Temperature scaling on the validation set.
    Learns a single scalar T: y_cal = sigmoid(logit / T).
    Does not alter ranking order; minimises NLL on val predictions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1, ROC-AUC, PR-AUC.

    If threshold is None the optimal F1 threshold on the data is used.
    """
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    prec_curve, rec_curve, thresholds = precision_recall_curve(y_true, y_score)
    f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-8)

    if threshold is None:
        best_idx = int(np.argmax(f1_curve[:-1]))
        threshold = float(thresholds[best_idx])

    y_pred = (y_score >= threshold).astype(int)
    precision = float(prec_curve[:-1][best_idx] if threshold is None
                      else np.mean(y_pred[y_true == 1] == 1) if y_pred.sum() > 0 else 0.0)
    recall = float(rec_curve[:-1][best_idx] if threshold is None
                   else np.sum((y_pred == 1) & (y_true == 1)) / max(y_true.sum(), 1))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
    }


def compute_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute MRR and Hits@K for link-prediction ranking.

    Args:
        scores: (N,) predicted scores (higher = more likely positive)
        labels: (N,) binary ground-truth labels
        k_values: list of K values for Hits@K

    Returns:
        Dictionary with mrr, hits@1, hits@5, hits@10.
    """
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    # MRR: reciprocal rank of the first true positive
    pos_ranks = np.where(sorted_labels == 1)[0] + 1  # 1-indexed
    mrr = float(np.mean(1.0 / pos_ranks)) if len(pos_ranks) > 0 else 0.0

    results: Dict[str, float] = {"mrr": mrr}
    for k in k_values:
        hits = float(sorted_labels[:k].sum()) / max(labels.sum(), 1)
        results[f"hits@{k}"] = hits

    return results


def evaluate_subgroup(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    group_labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Stratified evaluation across demographic or clinical subgroups.

    Used to assess differential performance across patient subgroups
    (e.g. gender, age group) as a check on cumulative method bias.

    Args:
        y_true:       (N,) ground-truth labels
        y_score:      (N,) predicted scores
        groups:       (N,) integer group assignments
        group_labels: optional human-readable names for each group

    Returns:
        Per-group metric dictionaries.
    """
    unique_groups = np.unique(groups)
    results: Dict[str, Dict[str, float]] = {}

    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 2 or y_true[mask].sum() == 0:
            continue
        name = group_labels[int(g)] if group_labels is not None else str(g)
        results[name] = compute_classification_metrics(y_true[mask], y_score[mask])

    return results


# ─── Temperature Scaling ─────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration via temperature scaling.

    Learns a single scalar T on the validation set:
        y_cal = sigmoid(logit / T)

    T is optimised to minimise negative log-likelihood on validation
    predictions while leaving the ranking order unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.temperature.clamp(min=1e-3))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 1e-2,
        max_iter: int = 500,
    ) -> float:
        """Fit temperature on validation logits/labels. Returns learned T."""
        self.train()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits / self.temperature.clamp(min=1e-3), labels.float()
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()
        return float(self.temperature.item())
