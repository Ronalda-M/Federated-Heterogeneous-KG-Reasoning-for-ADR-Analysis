"""
Training objective for typed Drug–ADR link prediction.

Class-imbalance-aware weighted binary cross-entropy with type-aware
negative sampling:

    L = - sum_{(d,a) in E+} w_{d,a} log y_hat_{d,a}
        - sum_{(d,a) in E-} log(1 - y_hat_{d,a})

    w_{d,a} = softplus(gamma1 * conf_{d,a} + gamma2 * recency_{d,a})

Negatives are drawn via:
    1. Type-constrained sampling — random Drug--ADR pairs that are
       structurally valid but unobserved in the KG.
    2. Hard negatives — drug--ADR pairs that co-occur in the same
       document or report (co-mentioned) but have no confirmed causal
       link in the KG.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Class-imbalance-aware binary cross-entropy.

    Per-positive weight:
        w_{d,a} = softplus(gamma1 * conf + gamma2 * recency)

    where conf is the evidence confidence score (e.g. normalised
    report count) and recency is the temporal weight of the most
    recent supporting report.
    """

    def __init__(
        self,
        gamma1: float = 1.0,
        gamma2: float = 0.5,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.label_smoothing = label_smoothing

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        conf: Optional[torch.Tensor] = None,
        recency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pos_scores: (N_pos,) raw logits for positive pairs
            neg_scores: (N_neg,) raw logits for negative pairs
            conf:       (N_pos,) evidence confidence scores
            recency:    (N_pos,) recency weights

        Returns:
            Scalar loss.
        """
        # Per-positive sample weight
        if conf is not None and recency is not None:
            w = F.softplus(self.gamma1 * conf + self.gamma2 * recency)
        else:
            w = torch.ones_like(pos_scores)

        # Positive loss with label smoothing
        pos_target = torch.ones_like(pos_scores) * (1.0 - self.label_smoothing)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, pos_target, weight=w, reduction="sum"
        )

        # Negative loss
        neg_target = torch.zeros_like(neg_scores)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, neg_target, reduction="sum"
        )

        total = pos_scores.numel() + neg_scores.numel()
        return (pos_loss + neg_loss) / max(total, 1)


class TypeAwareNegativeSampler:
    """
    Samples negative Drug--ADR pairs for training.

    Two strategies:
        type_constrained  — uniform random sampling restricted to valid
                            Drug--ADR node-type pairs (no cross-type noise)
        hard              — co-mentioned pairs with no confirmed KG edge
    """

    def __init__(
        self,
        num_drugs: int,
        num_adrs: int,
        hard_negative_ratio: float = 0.3,
        negatives_per_positive: int = 5,
        seed: int = 42,
    ) -> None:
        self.num_drugs = num_drugs
        self.num_adrs = num_adrs
        self.hard_ratio = hard_negative_ratio
        self.k = negatives_per_positive
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        # Co-mentioned pairs are populated externally from the KG
        self._hard_pool: Optional[torch.Tensor] = None

    def register_hard_negatives(self, pairs: torch.Tensor) -> None:
        """
        Register pre-computed co-mentioned-but-unconfirmed drug--ADR pairs.

        Args:
            pairs: (M, 2) tensor of [drug_idx, adr_idx] hard-negative pairs
        """
        self._hard_pool = pairs

    def sample(
        self,
        pos_drug_idx: torch.Tensor,
        pos_adr_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns negative drug/ADR index tensors of size (N_pos * k,).
        """
        n = pos_drug_idx.size(0)
        n_hard = int(n * self.k * self.hard_ratio)
        n_random = n * self.k - n_hard

        # Type-constrained random negatives
        rand_drug = torch.randint(0, self.num_drugs, (n_random,),
                                   generator=self.rng)
        rand_adr = torch.randint(0, self.num_adrs, (n_random,),
                                  generator=self.rng)

        if n_hard > 0 and self._hard_pool is not None:
            idx = torch.randint(0, self._hard_pool.size(0), (n_hard,),
                                generator=self.rng)
            hard = self._hard_pool[idx]
            neg_drug = torch.cat([rand_drug, hard[:, 0]])
            neg_adr = torch.cat([rand_adr, hard[:, 1]])
        else:
            neg_drug = rand_drug
            neg_adr = rand_adr

        return neg_drug, neg_adr
