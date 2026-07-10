"""
Drug–ADR scoring decoder.

Single-drug scoring:
    s(d, a) = z_d^T R_causes z_a + psi(phi_e(x_{(d,a)}))
    y_hat_{d,a} = sigmoid(s(d, a))

Multi-drug (polypharmacy) scoring:
    s({d_1, ..., d_K}, a) = sum_k s(d_k, a)
                           + sum_{k < k'} psi_int(z_{d_k}, z_{d_{k'}}, z_a)
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearDecoder(nn.Module):
    """
    Relation-specific bilinear decoder for typed link prediction.

    Scores candidate triple (d, DRUG_CAUSES_AE, a) as:
        s(d, a) = z_d^T R_causes z_a + psi(phi_e(x_{(d,a)}))

    where psi is a small MLP applied to the edge provenance attributes.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        edge_mlp_hidden: int = 64,
        edge_mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # R_causes: learnable relation matrix
        self.R_causes = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim)
        )
        nn.init.xavier_uniform_(self.R_causes)

        # psi: edge-feature MLP
        self.use_edge_feat = edge_feat_dim > 0
        if self.use_edge_feat:
            layers: List[nn.Module] = []
            dims = [edge_feat_dim] + [edge_mlp_hidden] * (edge_mlp_layers - 1) + [1]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.GELU())
            self.psi = nn.Sequential(*layers)

    def forward(
        self,
        z_d: torch.Tensor,
        z_a: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_d:        (B, D) drug embeddings
            z_a:        (B, D) ADR embeddings
            edge_attr:  (B, F) edge provenance features (optional)

        Returns:
            scores: (B,) raw logits before sigmoid
        """
        # Bilinear term: z_d^T R z_a
        bilinear = (z_d @ self.R_causes * z_a).sum(dim=-1)  # (B,)

        if self.use_edge_feat and edge_attr is not None:
            edge_score = self.psi(edge_attr).squeeze(-1)      # (B,)
            return bilinear + edge_score

        return bilinear

    def predict(
        self,
        z_d: torch.Tensor,
        z_a: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns calibrated probabilities via sigmoid."""
        return torch.sigmoid(self.forward(z_d, z_a, edge_attr))


class PolypharmacyDecoder(nn.Module):
    """
    Extends BilinearDecoder to multi-drug (polypharmacy) settings.

    s({d_1,...,d_K}, a) = sum_k s(d_k, a)
                        + sum_{k < k'} psi_int(z_{d_k}, z_{d_{k'}}, z_a)

    The pairwise interaction term psi_int captures synergistic or
    antagonistic effects between concurrently administered drugs with
    respect to a candidate adverse reaction.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int = 0,
        edge_mlp_hidden: int = 64,
        interaction_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.single = BilinearDecoder(
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            edge_mlp_hidden=edge_mlp_hidden,
        )

        # psi_int: bilinear MLP over drug-pair embeddings conditioned on ADR
        self.psi_int = nn.Sequential(
            nn.Linear(hidden_dim * 3, interaction_hidden),
            nn.GELU(),
            nn.Linear(interaction_hidden, 1),
        )

    def forward(
        self,
        z_drugs: List[torch.Tensor],
        z_a: torch.Tensor,
        edge_attrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_drugs:    list of K drug embedding tensors, each (B, D)
            z_a:        (B, D) ADR embeddings
            edge_attrs: optional list of K edge attribute tensors

        Returns:
            scores: (B,) polypharmacy risk scores
        """
        # Sum of single-drug scores
        score = torch.zeros(z_a.size(0), device=z_a.device)
        for k, z_d in enumerate(z_drugs):
            ea = edge_attrs[k] if (edge_attrs is not None) else None
            score = score + self.single(z_d, z_a, ea)

        # Pairwise interaction terms
        for k, kp in combinations(range(len(z_drugs)), 2):
            inp = torch.cat([z_drugs[k], z_drugs[kp], z_a], dim=-1)  # (B, 3D)
            score = score + self.psi_int(inp).squeeze(-1)

        return score

    def predict(
        self,
        z_drugs: List[torch.Tensor],
        z_a: torch.Tensor,
        edge_attrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(z_drugs, z_a, edge_attrs))
